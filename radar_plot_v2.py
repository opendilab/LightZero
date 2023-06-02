import numpy as np
import matplotlib.pyplot as plt

# 评分数据
data = {
    "AlphaZero": [2, 0, 0, 1, 2, 1],
    "MuZero": [2, 0, 2, 2, 2, 1],
    "EfficientZero": [3, 0, 3, 3, 3, 1],
    "Sampled MuZero": [3, 4, 3, 3, 3, 1],
    "Gumbel MuZero": [3, 0, 3, 3, 4, 1],
    "Stochastic MuZero": [3, 0, 4, 3, 3, 1],
}

# 设置能力标签
labels = ["Discrete Action Space", "Continuous Action Space", "Stochastic Dynamics", "Simulator Inaccessible", "High Simulation Cost", "Hard Exploration"]

num_vars = len(labels)

# 计算角度
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# 使图形闭合
angles += angles[:1]
for algo, scores in data.items():
    scores += scores[:1]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

# 画线
colors = ["b", "b", "g", "g", "g", "r"]
for algo, scores in data.items():
    ax.plot(angles, scores, label=algo)
    ax.fill(angles, scores, alpha=0.25)

# 设置角度和标签
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), labels)

# 设置雷达图的范围
ax.set_ylim(0, 4)

# 添加图例
# ax.legend(loc='center left', bbox_to_anchor=(-0.5, 0.5))
ax.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))

# 修改雷达图样式
ax.spines["polar"].set_visible(False)
ax.set_rticks([0, 1, 2, 3, 4])
ax.set_yticklabels([0, 1, 2, 3, 4])

# 设置标签颜色
label_colors = ["b", "b", "g", "g", "g", "r"]
for i, label in enumerate(ax.get_xticklabels()):
    label.set_color(label_colors[i])
    label.set_fontsize(11)

# plt.tight_layout()

# 选择保存格式
save_format = "pdf"  # 可选 "pdf" 或 "png"

if save_format == "pdf":
    plt.savefig("radar_lightzero.pdf", bbox_inches="tight")
elif save_format == "png":
    plt.savefig("radar_lightzero.png", bbox_inches="tight")

# plt.show()