# run_pong_qwen_ddp.py
import os, re, json, random
from dataclasses import dataclass
from collections import deque, namedtuple
from typing import List, Tuple, Union
import numpy as np
import shutil
from PIL import Image
import torch
import torch.distributed as dist

from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration

from easydict import EasyDict
from zoo.atari.envs.atari_lightzero_env import AtariEnvLightZero


def to_model_image(arr: Union[np.ndarray, torch.Tensor], channel_last: bool, use_pil: bool):
    """
    返回：
      - use_pil=True  -> PIL.Image(RGB)
      - use_pil=False -> numpy HWC uint8
    """
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr)

    # 2D 灰度 -> HWC
    if arr.ndim == 2:
        arr = arr[:, :, None]

    # 统一到 HWC
    if channel_last:
        hwc = arr
    else:
        assert arr.ndim == 3 and arr.shape[0] in (1, 3), f"Expect (C,H,W) or (H,W,C), got {arr.shape}"
        hwc = np.transpose(arr, (1, 2, 0))

    # 灰度扩 3 通道
    if hwc.shape[-1] == 1:
        hwc = np.repeat(hwc, 3, axis=-1)

    # 归一到 uint8
    if hwc.dtype != np.uint8:
        if hwc.max() <= 1.0:
            hwc = hwc * 255.0
        hwc = np.clip(hwc, 0, 255).astype(np.uint8)

    if use_pil:
        return Image.fromarray(hwc, mode="RGB")
    else:
        return hwc



def init_distributed():
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # 设定 device
    local_rank = int(os.getenv("LOCAL_RANK", rank % max(1, torch.cuda.device_count())))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


Transition = namedtuple("Transition", ["step", "image", "action_str"])

class QwenPongPolicy:
    """
    - 历史 n 帧（仅包含：图像 + 我们当时的动作字符串）
    - 指令结构（中文提示语义一致，英文更利于指令稳定）：
        环境描述 + 任务描述 + 当前图片 + <image> + 可选动作（字符串列表）
        + 历史轨迹（只含 历史图片 + 历史动作字符串）
      要求模型输出：单行 纯动作字符串（如 RIGHTFIRE）
    - 解析失败则从 allowed 随机抽取一个字符串，再映射回动作 id
    - 支持 FlashAttention-2（若不可用自动回退）
    """
    # 6 个官方动作名
    ID2NAME = {
        0: "NOOP",
        1: "FIRE",
        2: "RIGHT",
        3: "LEFT",
        4: "RIGHTFIRE",
        5: "LEFTFIRE",
    }
    NAME2ID = {v: k for k, v in ID2NAME.items()}

    ACTION_EXPLAIN = {
        "NOOP": "Do nothing (stay still).",
        "FIRE": "Serve a new point(use only at the start of a rally).",
        "RIGHT": "Move your RIGHT paddle UP in this Pong port.",
        "LEFT": "Move your RIGHT paddle DOWN in this Pong port.",
        "RIGHTFIRE": "Move UP and SERVE simultaneously (use only to start a rally).",
        "LEFTFIRE": "Move DOWN and SERVE simultaneously (use only to start a rally).",
    }


    def __init__(self, model_name: str, dtype: torch.dtype, history_n: int,
                 use_pil: bool, channel_last: bool, device: torch.device, save_dir: str = "pong_ddp_frames", save_image=False, rank: int = 0, topk: int = 1):
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map={"": device.index},
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )
        
        self.model.eval()

        self.history_n = history_n
        self.buffer: deque[Transition] = deque(maxlen=history_n)
        self.use_pil = use_pil
        self.channel_last = channel_last
        self.device = device
        self.save_image = save_image
        self.save_dir = save_dir
        self.rank = rank
        self.rank_dir = os.path.join(self.save_dir, f"rank{rank:02d}")
        self.topk = topk
        if os.path.exists(self.rank_dir):
            shutil.rmtree(self.rank_dir)

        os.makedirs(self.rank_dir, exist_ok=True)
        self.meta_path = os.path.join(self.rank_dir, "trajectory.jsonl")

    def save_pil_if_enabled(self, img: Image.Image, save_root: str, step: int):
        d = os.path.join(save_root, f"rank{self.rank:02d}")
        os.makedirs(d, exist_ok=True)
        img.save(os.path.join(d, f"frame_{step:06d}.png"))
    
    
    def log_step(self, step: int, action_id: int, action_str: str, reward: float, topk_seq=None):
        rec = {
            "step": int(step),
            "action_id": int(action_id),
            "action": str(action_str),
            "reward": float(reward),
        }
        if topk_seq is not None:
            rec["topk_seq"] = topk_seq       # 新增：不受约束的 top-k 序列及概率
        with open(self.meta_path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


    def _build_messages_and_images(self, cur_img, allowed_names: List[str]):
        """
        user.content 顺序（按你的要求）：
        1) 环境描述 + 任务描述（文本）
        2) 当前图片 <image>
        3) 可选动作（字符串列表）+ 对这 6 个动作的清晰解释
        4) 历史轨迹（只包含：历史图片 + 对应动作字符串）
        5) 输出格式要求：只返回一行 {ACTION: <action_str>}
        """
        content = []
        images_for_processor = []

        # 1) 环境 + 任务
        content.append({
            "type": "text",
            "text": (
                "Environment: Atari Pong (ALE) — two paddles rally a ball.\n"
                "Task: You control the right green paddle. Keep the paddle aligned vertically with the ball to return the ball and avoid losing it. The left paddle will hit the white ball, and you should try to land the white ball in the center of the right green paddle to return the ball and score.\n"
            )
        })

        # 2) 当前图片
        content.append({"type": "text", "text": "Current state image:"})
        content.append({"type": "image", "image": cur_img})
        images_for_processor.append(cur_img)

        # 3) 可选动作 + 解释
        allowed_str = ", ".join(allowed_names)
        # 解释文本（只针对当前允许的动作给出说明）
        explain_lines = []
        for name in allowed_names:
            desc = self.ACTION_EXPLAIN.get(name, "")
            if desc:
                explain_lines.append(f"- {name}: {desc}")
        explain_text = "\n".join(explain_lines)

        content.append({
            "type": "text",
            "text": (
                f"Available actions (choose exactly one string): {allowed_str}\n"
                # "Action semantics:\n"
                # f"{explain_text}\n"
                "Heuristic (to guide your choice): if the ball is above your paddle, choose an RIGHT action; "
                "if the ball is below, choose a LEFT action; if perfectly aligned and rally is active, NOOP briefly is acceptable."
            )
        })
        
        # 4) 历史交互轨迹（只包含：历史图片 + 当时选择的动作字符串）
        if len(self.buffer) > 0:
            # 在输出历史之前，先加一句说明
            content.append({"type": "text", "text":
                f"Now you are at step t. The following shows the previous {len(self.buffer)} steps of history (from most recent to oldest):\n"
            })

            # 然后再逐条列出历史轨迹
            for k, tr in enumerate(list(self.buffer)[::-1], start=1):  # k=1 表示 t−1
                content.append({"type": "text", "text": f"(t−{k}) observation:"})
                content.append({"type": "image", "image": tr.image})
                content.append({"type": "text", "text": f"(t−{k}) you chose: {tr.action_str}\n"})
                images_for_processor.append(tr.image)

        # 5) 输出格式要求（只返回一行 {ACTION: <action_str>}）
        content.append({
            "type": "text",
            "text": (
                "\nOutput requirement:\n"
                "- Return EXACTLY ONE line in the form: {ACTION: <action_str>}\n"
                f"- <action_str> MUST be one of: {allowed_str}\n"
            )
        })

        messages = [
            {"role": "system", "content": "You are a precise action selector for Atari Pong. Always follow the requested output format."},
            {"role": "user", "content": content},
        ]
        return messages, images_for_processor

    def _parse_action_string(self, text: str, allowed_names: List[str]) -> str:
        # 为避免 RIGHTFIRE 被 RIGHT 抢先匹配，按长度降序
        names_sorted = sorted(allowed_names, key=len, reverse=True)

        alt = "|".join(map(re.escape, names_sorted))
        pattern = rf"""\{{\s*"?ACTION"?\s*[:：]\s*"?\s*({alt})\s*"?\s*\}}"""

        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()

        return random.choice(allowed_names)

    @torch.inference_mode()
    def decide(self, obs_dict: dict, step: int):
        allowed_ids = [i for i, v in enumerate(obs_dict.get("action_mask", [1]*6)) if int(v) == 1]
        allowed_names = [self.ID2NAME[i] for i in allowed_ids]

        cur_img = to_model_image(obs_dict["observation"], channel_last=False, use_pil=self.use_pil)

        messages, images_for_processor = self._build_messages_and_images(cur_img, allowed_names)
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = self.processor(
            text=prompt,
            images=images_for_processor,
            return_tensors="pt"
        ).to(self.device)

        tok = self.processor.tokenizer
        gen_out = self.model.generate(
            **inputs,
            max_new_tokens=16,
            num_beams=self.topk,
            num_return_sequences=self.topk,
            do_sample=False,                 # 纯测概率，不采样
            output_scores=True,
            return_dict_in_generate=True,
            length_penalty=1.0,              # 不做长度归一
            early_stopping=True,
            eos_token_id=getattr(tok, "eos_token_id", None),
            pad_token_id=getattr(tok, "pad_token_id", getattr(tok, "eos_token_id", None)),
        )

        prompt_len = int(inputs["input_ids"].shape[1])
        seq_new = gen_out.sequences[:, prompt_len:]
        decoded = self.processor.tokenizer.batch_decode(seq_new, skip_special_tokens=True)
        decoded = [s.strip() for s in decoded]

        trans = self.model.compute_transition_scores(
            gen_out.sequences, gen_out.scores, gen_out.beam_indices, normalize_logits=True
        )  # shape: [num_return, gen_steps]
        gen_steps = len(gen_out.scores)
        logps = trans[:, -gen_steps:].sum(dim=1)  # 每条序列的 log P

        # 在 top-k 集合里归一化成概率
        probs = torch.softmax(logps, dim=0).tolist()

        # 组装 top-3（序列 + 概率）
        topk_seq = [{"text": txt, "prob": float(p)} for txt, p in zip(decoded, probs)]

        # 仍然用你已有的正则从“top-1 文本”解析动作作为实际执行的动作
        out_text = decoded[0]
        action_str = self._parse_action_string(out_text, allowed_names)
        action_id = self.NAME2ID.get(action_str, random.choice(allowed_ids))

        if self.use_pil and self.save_image:
            self.save_pil_if_enabled(cur_img, self.save_dir, step)

        # 返回 topk_seq，供日志写入
        return action_id, action_str, out_text, topk_seq

    def record(self, prev_obs: dict, action_id: int, step: int):
        img = to_model_image(prev_obs["observation"], channel_last=False, use_pil=self.use_pil)
        action_str = self.ID2NAME[action_id]
        self.buffer.append(Transition(step=step, image=img, action_str=action_str))


if __name__ == "__main__":
    rank, world_size, local_rank = init_distributed()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    base_seed = 12345
    random.seed(base_seed + rank)
    np.random.seed(base_seed + rank)
    torch.manual_seed(base_seed + rank)

    config = EasyDict(dict(
        collector_env_num=8,
        evaluator_env_num=3,
        n_evaluator_episode=3,
        env_id='PongNoFrameskip-v4',
        env_type='Atari',
        observation_shape=[3, 96, 96],
        collect_max_episode_steps=int(1.08e5),
        eval_max_episode_steps=int(1.08e5),
        gray_scale=False,
        frame_skip=4,
        frame_stack_num=1,
        episode_life=True,
        clip_rewards=True,
        channel_last=False,
        render_mode_human=False,
        scale=True,
        warp_frame=True,
        save_video=False,
        transform2string=False,
        game_wrapper=True,
        stop_value=int(1e6),
        save_replay=False,
        replay_path=None,
    ))
    config.max_episode_steps = config.eval_max_episode_steps
    env = AtariEnvLightZero(config)

    policy = QwenPongPolicy(
        model_name="/fs-computility/niuyazhe/shared/xiongjyu/model/Qwen2.5-VL-3B-Instruct",
        dtype=torch.bfloat16,
        history_n=2,
        use_pil=True,
        channel_last=config.channel_last,
        device=device,
        save_dir="/fs-computility/niuyazhe/shared/xiongjyu/jericho/LightZero/pong_ddp_frames",
        save_image=True,
        rank=rank,
        topk=3
    )

    obs = env.reset()
    episode_return, steps = 0.0, 0

    while True:
        action_id, action_str, raw, topk_seq = policy.decide(obs, step=steps)
        prev_obs = obs
        obs, reward, done, info = env.step(action_id)
        policy.log_step(steps, action_id, action_str, reward, topk_seq=topk_seq)

        policy.record(prev_obs, action_id, step=steps)

        episode_return += float(reward)
        steps += 1

        if done or steps >= config.max_episode_steps:
            print(f"[RANK {rank}/{world_size}] return={episode_return}, steps={steps}, info={info}")
            break
