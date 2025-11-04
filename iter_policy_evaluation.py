import numpy as np
import matplotlib.pyplot as plt

def setup_matplotlib_for_chinese():
    """
    为 matplotlib 设置中文字体，以解决中文显示为方块的问题。
    此函数会尝试为 macOS, Windows, 和 Linux 设置合适的字体。
    """
    try:
        # 优先选择 macOS 的苹方字体
        plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'SimHei']
    except Exception:
        # 如果失败，可能是其他系统，继续尝试
        try:
            # Windows 的黑体或微软雅黑
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        except Exception:
            # Linux 的文泉驿正黑
            try:
                plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
            except Exception:
                print("警告：未能找到可用的中文字体。图表中的中文可能无法正常显示。")
    
    # 解决负号'-'显示为方块的问题
    plt.rcParams['axes.unicode_minus'] = False
    print("Matplotlib 中文字体设置完成。")


def calculate_policy_evaluation():
    """
    使用迭代策略评估解决小格子世界问题。
    
    返回:
        V (np.ndarray): 最终收敛的价值函数网格。
        iterations_history (list): 每次迭代的序号列表。
        deltas_history (list): 每次迭代的 delta 值列表。
    """
    # --- 1. 环境设置 ---
    grid_size = 4
    V = np.zeros((grid_size, grid_size))
    gamma = 1.0
    reward = -1.0
    theta = 1e-6 # 收敛阈值

    # --- 2. 用于可视化的数据收集 ---
    iterations_history = []
    deltas_history = []
    
    iteration = 0
    while True:
        iteration += 1
        delta = 0
        V_old = V.copy()

        for i in range(grid_size):
            for j in range(grid_size):
                if (i == 0 and j == 0) or (i == grid_size - 1 and j == grid_size - 1):
                    continue

                # --- 3. 应用贝尔曼期望方程 ---
                current_v = 0
                
                # 动作: 上
                next_i_up = i - 1 if i > 0 else i
                current_v += 0.25 * (reward + gamma * V_old[next_i_up, j])

                # 动作: 下
                next_i_down = i + 1 if i < grid_size - 1 else i
                current_v += 0.25 * (reward + gamma * V_old[next_i_down, j])

                # 动作: 左
                next_j_left = j - 1 if j > 0 else j
                current_v += 0.25 * (reward + gamma * V_old[i, next_j_left])

                # 动作: 右
                next_j_right = j + 1 if j < grid_size - 1 else j
                current_v += 0.25 * (reward + gamma * V_old[i, next_j_right])

                V[i, j] = current_v
                delta = max(delta, abs(V[i, j] - V_old[i, j]))

        # --- 4. 记录历史并检查收敛 ---
        iterations_history.append(iteration)
        deltas_history.append(delta)
        
        if iteration % 10 == 0 or delta < theta:
            print(f"迭代 {iteration}: 最大变化量 (delta) = {delta:.8f}")

        if delta < theta:
            break
            
    return V, iterations_history, deltas_history

def print_value_grid(V):
    """以可读的网格格式打印最终的价值函数。"""
    print("\n--- 迭代策略评估已收敛! ---")
    print("最终价值函数 V(s):")
    
    grid_size = V.shape[0]
    print("+------------------------------------------------+")
    for i in range(grid_size):
        row_str = "| "
        for j in range(grid_size):
            row_str += f"{V[i, j]:>8.3f} | "
        print(row_str)
        print("+------------------------------------------------+")

def save_delta_plot_as_image(iterations, deltas, filename="delta_convergence_curve.png"):
    """
    使用 matplotlib 生成 delta 收敛曲线图，并将其保存为图像文件。

    参数:
        iterations (list): 迭代次数的列表。
        deltas (list): delta 值的列表。
        filename (str): 保存图像的文件名。
    """
    print(f"\n正在生成图表并保存为 '{filename}'...")
    
    # 1. 创建一个图形和一个坐标轴
    plt.figure(figsize=(10, 6), dpi=100)
    
    # 2. 绘制数据
    plt.plot(iterations, deltas, label='Delta (价值函数最大变化量)', color='dodgerblue', linewidth=2)
    
    # 3. 设置 Y 轴为对数刻度
    plt.yscale('log')
    
    # 4. 添加标题和轴标签
    plt.title('迭代策略评估的收敛过程', fontsize=16)
    plt.xlabel('迭代次数 (Iteration)', fontsize=12)
    plt.ylabel('Delta (对数刻度)', fontsize=12)
    
    # 5. 添加图例
    plt.legend()
    
    # 6. 添加网格以便于阅读
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    
    # 7. 保存图像
    try:
        plt.savefig(filename, bbox_inches='tight')
        print(f"图表已成功保存为 '{filename}'")
    except Exception as e:
        print(f"保存图表时出错: {e}")
    
    # 8. 关闭图形以释放内存
    plt.close()


if __name__ == '__main__':
    # 在所有绘图操作之前，首先设置好字体环境
    setup_matplotlib_for_chinese()
    
    # 1. 执行计算
    final_V, iterations_hist, deltas_hist = calculate_policy_evaluation()
    
    # 2. 在控制台打印最终的价值网格
    print_value_grid(final_V)
    
    # 3. 生成并保存 delta 曲线图
    save_delta_plot_as_image(iterations_hist, deltas_hist, "delta_convergence_curve.png")