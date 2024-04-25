"""
这段代码定义了一个名为 GSM8kConfig 的类，继承自 SearchConfig 类。这个类的主要目的是为 GSM8k（一个数学问题数据集）配置搜索参数和相关的提示（prompt）。

类中的方法主要有以下几个：

__init__: 初始化方法，用于设置各种参数的默认值。
update_example: 更新示例，并根据提示（prompt）更新相关的属性。
get_actions: 根据当前状态生成可能的操作（action）。
fast_reward: 根据状态和操作快速计算奖励。
calculate_reward: 计算最终奖励。
reward: 计算给定状态和操作的奖励。
"""
import io
import re
from typing import TypedDict, Optional
import numpy as np

from world_model import GSM8kState, GSM8kAction, GSM8kPromptDict
from reasoners import SearchConfig, LanguageModel

class GSM8kUsefulPrompt(TypedDict):
    input: str
    question_prefix: str
    subquestion_prefix: str
    new_subquestion_prefix: str
    useful_prefix: str

class GSM8kConfig(SearchConfig):
    def __init__(self,
                 base_model: LanguageModel,
                 useful_prompt: GSM8kUsefulPrompt,
                 n_actions=4,
                 batch_size=1,
                 temperature=0.8,
                 top_k=50,
                 top_p=0.95,
                 reward_alpha=0.5,
                 reward_confidence_default=0.8,
                 depth_limit=5,
                 force_terminating_on_depth_limit=True,
                 force_overall_prompt_on_overall_question=True,
                 force_overall_question_on_overall_prompt=True) -> None:
        super().__init__()
        self.base_model = base_model  # 基础语言模型
        self.useful_prompt = useful_prompt  # 有用的提示
        self.example = ''  # 示例
        self.batch_size = batch_size  # 批次大小
        self.temperature = temperature  # 温度参数
        self.top_k = top_k  # top_k 采样参数
        self.top_p = top_p  # top_p 采样参数
        self.n_actions = n_actions  # 操作数量
        self.force_terminating_on_depth_limit = force_terminating_on_depth_limit  # 是否在深度限制时强制终止
        self.depth_limit = depth_limit  # 深度限制
        self.reward_alpha = reward_alpha  # 奖励的 alpha 参数
        self.reward_confidence_default = reward_confidence_default  # 默认奖励置信度
        self.force_overall_prompt_on_overall_question = force_overall_prompt_on_overall_question  # 是否在整体问题上强制使用整体提示
        self.force_overall_question_on_overall_prompt = force_overall_question_on_overall_prompt  # 是否在整体提示上强制使用整体问题
        self.overall_question: Optional[str] = None  # 整体问题
        self.prompt_examples = ""  # 提示示例
        self.n_shots = 0  # 示例数量

    def update_example(self, example: str, prompt: GSM8kPromptDict = None) -> None:
        super().update_example(example, prompt=prompt)  # 调用父类的 update_example 方法

        assert prompt is not None
        self.prompt = prompt  # 更新提示
        with io.StringIO() as f:
            f.write(self.prompt['instruction'] + '\n\n')  # 写入指令
            for idx, example in enumerate(self.prompt['interactive_examples']):
                f.write(example.format(idx=idx + 1) + '\n\n')  # 写入交互示例
            self.n_shots = len(self.prompt['interactive_examples'])  # 更新示例数量
            self.prompt_examples = f.getvalue()  # 更新提示示例

        if self.force_overall_prompt_on_overall_question or self.force_overall_question_on_overall_prompt:
            print(self.example)
            try:
                self.overall_question = re.match('.*((([A-Z].* (calculate|how|what|find|true or false))|((Calculate|How|What|Find|True or false))).*)$', self.example, flags=re.DOTALL | re.IGNORECASE)[1]  # 提取整体问题
            except Exception as e:
                import pdb;pdb.set_trace()
                print(e)

    def get_actions(self, state: GSM8kState, ) -> list[GSM8kAction]:
        with io.StringIO() as f:
            f.write(self.prompt_examples)  # 写入提示示例
            f.write(self.prompt["question_prefix"].format(idx=self.n_shots + 1, question=self.example) + "\n")  # 写入问题前缀
            for idx, (q, a, _) in enumerate(state):
                f.write(
                    self.prompt["subquestion_prefix"].format(idx=self.n_shots + 1, sub_idx=idx + 1) + " " + q + "\n")  # 写入子问题前缀和问题
                f.write(self.prompt["answer_prefix"].format(idx=self.n_shots + 1, sub_idx=idx + 1) + " " + a + "\n")  # 写入答案前缀和答案
            f.write(self.prompt["subquestion_prefix"].format(idx=self.n_shots + 1, sub_idx=len(state) + 1))  # 写入新的子问题前缀
            if at_depth_limit := self.force_terminating_on_depth_limit and len(state) + 1 >= self.depth_limit:
                f.write(" " + self.prompt["overall_question_prefix"])  # 如果达到深度限制，写入整体问题前缀
            model_input = f.getvalue()  # 获取模型输入

        n_actions = 1 if at_depth_limit else self.n_actions  # 如果达到深度限制，操作数量为 1，否则为 self.n_actions
        temperature = 0 if at_depth_limit else self.temperature  # 如果达到深度限制，温度为 0，否则为 self.temperature
        outputs = []
        for idx in range(0, n_actions, self.batch_size):
            n_samples = min(n_actions - idx, self.batch_size)
            outputs += self.base_model.generate([model_input] * n_samples,
                                                hide_input=True,
                                                do_sample=True,
                                                temperature=temperature,
                                                top_k=self.top_k,
                                                top_p=self.top_p,
                                                eos_token_id='\n').text  # 生成输出

        outputs = [output.strip() for output in outputs]  # 去除输出中的空白字符
        if at_depth_limit:
            outputs = [self.prompt["overall_question_prefix"] + ' ' + output for output in outputs]  # 如果达到深度限制，添加整体问题前缀

        """
        如果设置了 force_overall_question_on_overall_prompt,且输出中包含整体问题前缀,则用整体问题替换输出。
        如果设置了 force_overall_prompt_on_overall_question,且输出与整体问题相同,则添加整体问题前缀。
        去除重复的输出,保持顺序。
        """
        if self.force_overall_question_on_overall_prompt:
            for i, output in enumerate(outputs):
                if self.prompt["overall_question_prefix"] in output:
                    outputs[i] = self.prompt["overall_question_prefix"] + ' ' + self.overall_question  # 如果输出中包含整体问题前缀,则用整体问题替换
        if self.force_overall_prompt_on_overall_question:
            for i, output in enumerate(outputs):
                if self.overall_question.lower() == output.lower():
                    outputs[i] = self.prompt["overall_question_prefix"] + ' ' + self.overall_question  # 如果输出与整体问题相同,则添加整体问题前缀

        outputs = list(dict.fromkeys(outputs))  # 去除重复的输出,保持顺序
        return outputs

    def fast_reward(self, state: GSM8kState, action: GSM8kAction) -> tuple[float, dict]:
        """
        定义了 fast_reward 方法,用于快速计算给定状态和操作的奖励。具体步骤为:

        构造模型输入,包括有用提示的输入部分、问题前缀、示例、子问题和新子问题等。
        获取下一个标记的对数概率,计算 "Yes" 和 "No" 的概率。
        获取 "Yes" 的概率作为有用概率。
        调用 calculate_reward 方法计算快速奖励。
        """
        with io.StringIO() as f:
            f.write(self.useful_prompt["input"])  # 写入有用提示的输入部分
            f.write(self.useful_prompt["question_prefix"] + self.example + "\n")  # 写入问题前缀和示例
            for idx, (q, _, _) in enumerate(state):
                f.write(self.useful_prompt["subquestion_prefix"].format(idx + 1) + " " + q + "\n")  # 写入子问题前缀和问题
            f.write(self.useful_prompt["new_subquestion_prefix"].format(len(state) + 1) + " " + action + "\n")  # 写入新子问题前缀和操作
            f.write(self.useful_prompt["useful_prefix"])  # 写入有用前缀
            model_input = f.getvalue()  # 获取模型输入
        
        logits = self.base_model.get_next_token_logits(model_input, ["Yes", "No"])[0]  # 获取下一个标记的对数概率
        probs = np.exp(logits) / np.sum(np.exp(logits))  # 计算概率
        useful_prob = probs[0]  # 获取有用的概率
        fast_reward, _ = self.calculate_reward(useful_prob)  # 计算快速奖励
        return fast_reward, {'r_useful': useful_prob}  # 返回快速奖励和有用概率

    def calculate_reward(self, r_useful, r_conf=None):
        if r_conf is None:
            r_conf = self.reward_confidence_default  # 如果置信度为 None,则使用默认值
        return r_useful ** self.reward_alpha * r_conf ** (1 - self.reward_alpha), {'r_useful': r_useful,
                                                                                   'r_conf': r_conf}  # 计算奖励

    def reward(self, state: GSM8kState, action: GSM8kAction,
               r_useful: float = None,
               confidence: float = None) -> tuple[float, dict]:
        assert r_useful is not None, "useful_reward is required to calculate reward in this search config, consider passing it in fast_reward"  # 断言有用奖励不为 None
        assert confidence is not None, "confidence is required to calculate reward in this search config, consider passing it in world model's step"  # 断言置信度不为 None
        return self.calculate_reward(r_useful, confidence)  # 计算奖励