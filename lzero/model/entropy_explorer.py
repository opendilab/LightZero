import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os

class EntropyExplorer:
    """
    一个用于探查、计算和生成具有特定熵的离散概率分布的工具。
    现在支持将可视化结果保存为PNG文件，并解决了中文字体显示问题。
    """
    def __init__(self, action_space_size: int):
        """
        初始化探查器。
        
        参数:
        - action_space_size (int): 动作空间的维度 (例如，对于 torch.Size([9])，此值为 9)。
        """
        if not isinstance(action_space_size, int) or action_space_size <= 1:
            raise ValueError("action_space_size 必须是大于1的整数。")
            
        self.action_space_size = action_space_size
        self.min_entropy = 0.0
        self.max_entropy = torch.log(torch.tensor(self.action_space_size, dtype=torch.float32)).item()
        
        print(f"初始化 EntropyExplorer，动作空间大小为: {self.action_space_size}")
        print(f"该空间的最小熵为: {self.min_entropy:.4f}")
        print(f"该空间的最大熵 (均匀分布) 为: {self.max_entropy:.4f}\n")

    def calculate_entropy(self, policy_tensor: torch.Tensor, is_logits: bool = True) -> float:
        """
        计算给定策略张量的熵。
        """
        if policy_tensor.dim() != 1 or policy_tensor.shape[0] != self.action_space_size:
            raise ValueError(f"输入张量的形状必须是 torch.Size([{self.action_space_size}]), 但得到的是 {policy_tensor.shape}")

        if is_logits:
            distribution = Categorical(logits=policy_tensor)
        else:
            if not torch.allclose(policy_tensor.sum(), torch.tensor(1.0), atol=1e-6):
                print(f"警告: 输入的概率总和不为1 (总和为: {policy_tensor.sum().item()})，结果可能不准确。")
            distribution = Categorical(probs=policy_tensor)
            
        return distribution.entropy().item()

    def find_distribution_for_entropy(self, target_entropy: float, learning_rate: float = 0.01, num_steps: int = 1500, verbose: bool = False) -> np.ndarray:
        """
        通过优化找到一个具有特定目标熵的概率分布。
        """
        if not (self.min_entropy <= target_entropy <= self.max_entropy):
            raise ValueError(f"目标熵 {target_entropy:.4f} 超出有效范围 [{self.min_entropy:.4f}, {self.max_entropy:.4f}]。")

        logits = torch.randn(self.action_space_size, requires_grad=True)
        optimizer = optim.Adam([logits], lr=learning_rate)

        print(f"\n正在寻找熵为 {target_entropy:.4f} 的分布...")
        for step in range(num_steps):
            optimizer.zero_grad()
            current_entropy = Categorical(logits=logits).entropy()
            loss = (current_entropy - target_entropy).pow(2)
            loss.backward()
            optimizer.step()
            
            if verbose and (step % 200 == 0 or step == num_steps - 1):
                print(f"步骤 {step+1}/{num_steps} | 当前熵: {current_entropy.item():.4f} | 损失: {loss.item():.6f}")

        final_probs = F.softmax(logits, dim=-1)
        final_entropy = self.calculate_entropy(final_probs, is_logits=False)
        print(f"优化完成。最终分布的熵为: {final_entropy:.4f}")
        
        return final_probs.detach().numpy()

    def _set_chinese_font(self):
        """
        尝试设置一个支持中文的字体。
        """
        # 常见的支持中文的字体列表
        font_names = ['SimHei', 'Microsoft YaHei', 'Heiti TC', 'Arial Unicode MS', 'sans-serif']
        try:
            plt.rcParams['font.sans-serif'] = font_names
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
            print("中文字体已设置为 'SimHei' 或其他可用字体。")
        except Exception as e:
            print(f"设置中文字体失败: {e}。图表中的中文可能无法正常显示。")


    def visualize_distribution(self, probs: np.ndarray, title: str, save_dir: str = 'entropy_plots', filename: str = None):
        """
        使用条形图可视化概率分布，并将其保存为PNG文件。

        参数:
        - probs (np.ndarray): 要可视化的概率分布。
        - title (str): 图表标题。
        - save_dir (str): 保存图片的相对路径目录。
        - filename (str): 图片的文件名。如果为None，将不会保存。
        """
        self._set_chinese_font() # 设置中文字体

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 6))
        
        actions = np.arange(self.action_space_size)
        bars = ax.bar(actions, probs, color='deepskyblue', edgecolor='black', alpha=0.8)
        
        ax.set_title(title, fontsize=16, weight='bold')
        ax.set_xlabel("动作 (Action)", fontsize=12)
        ax.set_ylabel("概率 (Probability)", fontsize=12)
        ax.set_xticks(actions)
        ax.set_xticklabels([f'Action {i}' for i in actions])
        ax.set_ylim(0, max(np.max(probs) * 1.2, 0.2)) # 动态调整Y轴上限
        
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f'{yval:.3f}', ha='center', va='bottom', fontsize=9)
            
        plt.tight_layout()
        
        if filename:
            # 确保保存目录存在
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                print(f"已创建目录: '{save_dir}/'")
            
            full_path = os.path.join(save_dir, filename)
            
            # 保存图片
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"图表已成功保存到: '{full_path}'")
        else:
            plt.show() # 如果未提供文件名，则显示图片
            
        plt.close(fig) # 关闭图形，释放内存


if __name__ == '__main__':
    # --- 使用工具 ---
    
    # 1. 设置你的动作空间大小
    ACTION_SPACE = 9
    explorer = EntropyExplorer(action_space_size=ACTION_SPACE)
    
    # 定义保存图片的相对路径
    SAVE_DIRECTORY = "entropy_visualizations"

    # 2. 核心功能：为您的问题复现，生成熵为 2.15 的分布并保存
    target_entropy_high = 2.15
    generated_probs_high = explorer.find_distribution_for_entropy(target_entropy_high)
    
    # 详细恰当的命名
    filename_high = f"entropy_{target_entropy_high:.4f}_actions_{ACTION_SPACE}.png"
    title_high = f"熵约为 {target_entropy_high} 的一个可能分布 (高熵)"
    
    explorer.visualize_distribution(
        generated_probs_high, 
        title=title_high, 
        save_dir=SAVE_DIRECTORY,
        filename=filename_high
    )

    # 3. 更多示例，以建立直观感受
    
    # 示例 A: 中等熵
    target_entropy_mid = 1.6
    generated_probs_mid = explorer.find_distribution_for_entropy(target_entropy_mid)
    filename_mid = f"entropy_{target_entropy_mid:.4f}_actions_{ACTION_SPACE}.png"
    title_mid = f"熵约为 {target_entropy_mid} 的一个可能分布 (中熵)"
    explorer.visualize_distribution(
        generated_probs_mid,
        title=title_mid,
        save_dir=SAVE_DIRECTORY,
        filename=filename_mid
    )

    # 示例 B: 非常低的熵 (接近确定性)
    target_entropy_low = 0.2
    generated_probs_low = explorer.find_distribution_for_entropy(target_entropy_low)
    filename_low = f"entropy_{target_entropy_low:.4f}_actions_{ACTION_SPACE}.png"
    title_low = f"熵约为 {target_entropy_low} 的一个可能分布 (低熵)"
    explorer.visualize_distribution(
        generated_probs_low,
        title=title_low,
        save_dir=SAVE_DIRECTORY,
        filename=filename_low
    )