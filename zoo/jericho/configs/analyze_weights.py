import torch
import os
import numpy as np
from transformers import AutoModel
import torch.nn as nn

# ================= 配置路径 =================
# 你的 Checkpoint 路径
CKPT_PATH = "/mnt/shared-storage-user/puyuan/code/LightZero/zoo/jericho/configs/wm_best.pth.tar"
# 你的预训练模型路径 (训练时用的那个)
PRETRAINED_MODEL_PATH = "/mnt/shared-storage-user/puyuan/code/LightZero/pretrained_models/bge-base-en-v1.5"
# ===========================================

def clean_key(k):
    """清洗 key，去除编译或 DDP 产生的前缀"""
    return k.replace('_orig_mod.', '').replace('module.', '')

def check_weights_comprehensive():
    print(f"{'='*60}")
    print(f"🚀 全量权重一致性分析工具")
    print(f"{'='*60}")
    
    # 1. 加载 Checkpoint
    if not os.path.exists(CKPT_PATH):
        print(f"❌ 错误: 找不到 Checkpoint 文件: {CKPT_PATH}")
        return

    print(f"1. 正在加载 Checkpoint: {CKPT_PATH} ...")
    try:
        checkpoint = torch.load(CKPT_PATH, map_location='cpu')
        # 处理 LightZero/Dizoo 常见的嵌套结构
        if 'model' in checkpoint:
            raw_state_dict = checkpoint['model']
        elif 'policy' in checkpoint:
            raw_state_dict = checkpoint['policy']
        else:
            raw_state_dict = checkpoint
            
        # 预处理：清洗所有 Key (去除 _orig_mod 等干扰)
        state_dict = {clean_key(k): v for k, v in raw_state_dict.items()}
        print(f"   Checkpoint 加载成功，清洗后包含 {lete_dict)} 个键。")
    except Exception as e:
        print(f"❌ Checkpoint 加载失败: {e}")
        return

    # 2. 加载原始 HuggingFace 预训练模型 (基准)
    print(f"2. 正在加载原始 HF 模型: {PRETRAINED_MODEL_PATH} ...")
    try:
        orig_hf_model = AutoModel.from_pretrained(PRETRAINED_MODEL_PATH)
        hf_state_dict = orig_hf_model.state_dict()
        print(f"   原始 HF 模型加载成功，包含 {len(hf_state_dict)} 个参数张量。")
    except Exception as e:
        print(f"❌ 原始模型加载失败: {e}")
        return

    # ================= 验证环节 A: 自动定位前缀 =================
    print(f"\n{'='*20} 阶段 A: 定位 Backbone 前缀 {'='*20}")
    
    # 我们用一个特征极其明显的 Key 来定位：embeddings.word_embeddings.weight
    anchor_key = "embeddings.word_embeddings.weight"
    found_prefix = None
    
    # 在 Checkpoint 中寻找包含 anchor_key 的键
    for k in state_dict.keys():
        if k.endswith(anchor_key):
            # 提取前缀。例如 k = "representation_network.pretrained_model.embeddings..."
            # split 后取第一部分
            found_prefix = k.replace(anchor_key, "")
            break
            
    if found_prefix is None:
        print(f"❌ 致命错误: 在 Checkpoint 中完全找不到 HF 模型特征键 ({anchor_key})。")
        print("   可能原因: 模型结构完全不同，或者 Checkpoint 损坏。")
        return
    else:
        print(f"✅ 锁定前缀: '{found_prefix}'")
        print(f"   (这意味着 HF 的 key 'x' 在 Checkpoint 中对应 '{found_prefix}x')")

    # ================= 验证环节 B: 全量逐层对比 =================
    print(f"\n{'='*20} 阶段 B: 全量参数逐层对比 {'='*20}")
    
    stats = {
        "matched": 0,
        "mismatched_value": 0,
        "mismatched_shape": 0,
        "missing": 0,
        "total": len(hf_state_dict)
    }
    
    mismatch_details = []
    
    # 遍历 HF 原始模型的每一个 Key
    for hf_key, hf_tensor in hf_state_dict.items():
        # 构造 Checkpoint 中预期的 Key
        ckpt_target_key = found_prefix + hf_key
        
        if ckpt_target_key not in state_dict:
            stats["missing"] += 1
            mismatch_details.append(f"MISSING: {hf_key}")
            continue
            
        ckpt_tensor = state_dict[ckpt_target_key]
        
        # 1. 检查 Shape
        if ckpt_tensor.shape != hf_tensor.shape:
            stats["mismatched_shape"] += 1
            mismatch_details.append(f"SHAPE MISMATCH: {hf_key} (Ckpt: {ckpt_tensor.shape} vs HF: {hf_tensor.shape})")
            continue
            
        # 2. 检查数值 (统一转 float32 对比，忽略 fp16/fp32 精度微差)
        # 使用 1e-5 作为容忍度，因为 fp16 转换可能会引入微小误差
        diff = torch.abs(ckpt_tensor.float() - hf_tensor.float())
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        if max_diff > 1e-5:
            stats["mismatched_value"] += 1
            mismatch_details.append(f"VALUE DIFF: {hf_key} | MaxDiff: {max_diff:.6f} | MeanDiff: {mean_diff:.8f}")
        else:
            stats["matched"] += 1

    # ================= 输出报告 =================
    print(f"\n{'='*20} 最终分析报告 {'='*20}")
    print(f"总计检查参数量: {stats['total']}")
    print(f"✅ 完全一致: {stats['matched']}")
    print(f"❌ 数值不一致: {stats['mismatched_value']}")
    print(f"❌ 形状不一致: {stats['mismatched_shape']}")
    print(f"❓ 缺失参数:   {stats['missing']}")
    
    if stats["matched"] == stats["total"]:
        print(f"\n🎉 完美！Checkpoint 中的 Backbone 与原始 HF 模型完全一致。")
        print("   结论: 预训练权重加载正确，且在后续训练中被正确冻结 (Frozen)。")
    else:
        print(f"\n⚠️ 警告: 发现不一致！")
        
        if stats["mismatched_value"] > 0:
            print(f"   发现 {stats['mismatched_value']} 个层数值不同。")
            print("   可能原因: 训练时没有设置 `requires_grad=False`，导致 Backbone 被更新了。")
            print("   前 5 个不一致的层:")
            for detail in mismatch_details[:5]:
                print(f"     - {detail}")
                
        if stats["missing"] > 0:
            print(f"   发现 {stats['missing']} 个层在 Checkpoint 中丢失。")
            print("   可能原因: `strict=False` 加载时忽略了这些层，或者保存时过滤了。")

    # ================= 验证环节 C: 投影层 (Projection Head) =================
    # 这是一个额外的检查，确保非 HF 的部分（我们需要训练的部分）不是全 0
    print(f"\n{'='*20} 阶段 C: 验证可学习层 (Projection Head) {'='*20}")
    
    # 尝试几种常见的命名
    proj_candidates = ["embed_proj_head.weight", "projection_head.weight", "fc.weight"]
    proj_key = None
    
    # 基于之前找到的 prefix 回溯，通常 proj head 和 pretrained_model 是兄弟节点
    # 如果 prefix 是 "representation_network.pretrained_model."
    # 那么 proj head 可能是 "representation_network.embed_proj_head.weight"
    
    base_prefix = found_prefix.replace("pretrained_model.", "") # 移除 pretrained_model. 保留 representation_network.
    
    found_proj = False
    for cand in proj_candidates:
        target = base_prefix + cand
        if target in state_dict:
            w = state_dict[target]
            print(f"✅ 找到 Projection Layer: {target}")
            print(f"   - Shape: {w.shape}")
            print(f"   - Mean: {w.float().mean().item():.6f}")
            print(f"   - Std:  {w.float().std().item():.6f}")
            
            if torch.allclose(w, torch.zeros_like(w)):
                print("❌ 警告: 权重全为 0！模型可能未正确保存或初始化失败。")
            elif w.std() > 0:
                print("✅ 权重分布正常 (非全0，有方差)。")
            found_proj = True
            break
            
    if not found_proj:
        print(f"⚠️ 未找到常见的 Projection Head 命名 (尝试了 {proj_candidates})。请手动检查键名。")

if __name__ == "__main__":
    check_weights_comprehensive()