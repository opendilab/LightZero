import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from PIL import Image
from PIL import ImageDraw
from sklearn.manifold import TSNE



def visualize_reward_value_img_policy( original_images, reconstructed_images, target_predict_value, true_rewards,
                                      target_policy, predict_value, predict_rewards, predict_policy,
                                      not_plot_timesteps=[], suffix='pong', width=64
                                      ):
    # 确保输入张量的维度匹配
    assert original_images.shape[0] == reconstructed_images.shape[0] // original_images.shape[1]
    assert original_images.shape[1] == reconstructed_images.shape[0] // original_images.shape[0]
    assert original_images.shape[2:] == reconstructed_images.shape[1:]

    batch_size = original_images.shape[0]
    num_timesteps = original_images.shape[1]
    num_actions = predict_policy.shape[2]

    # 根据动作空间大小自适应颜色
    colors = plt.cm.viridis(np.linspace(0, 1, num_actions))

    for batch_idx in range(batch_size):
        # fig, ax = plt.subplots(3, 1, figsize=(20, 12), gridspec_kw={'height_ratios': [1, 1, 1]})
        fig, ax = plt.subplots(4, 1, figsize=(20, 12), gridspec_kw={'height_ratios': [1, 1, 1, 1]})

        # 绘制rewards和value的折线图
        timesteps = range(1, num_timesteps + 1)
        plot_timesteps = [t for t in timesteps if t not in not_plot_timesteps]

        true_rewards_filtered = [true_rewards[batch_idx, t - 1, 0].cpu().detach().numpy() for t in plot_timesteps]
        predict_rewards_filtered = [predict_rewards[batch_idx, t - 1, 0].cpu().detach().numpy() for t in
                                    plot_timesteps]
        predict_value_filtered = [predict_value[batch_idx, t - 1, 0].cpu().detach().numpy() for t in plot_timesteps]

        # ax[0].plot(plot_timesteps, true_rewards_filtered, 'g-', label='True Rewards')
        # ax[0].plot(plot_timesteps, predict_rewards_filtered, 'g--', label='Predict Rewards')
        # ax[0].plot(plot_timesteps, true_rewards_filtered, 'g-', label='True Rewards', marker='o')
        # ax[0].plot(plot_timesteps, predict_rewards_filtered, 'g--', label='Predict Rewards', marker='x')
        # ax[0].set_xticks(plot_timesteps)
        # # ax[0].set_xticklabels([])
        # ax[0].set_xticklabels(plot_timesteps)  # 添加时间步索引作为x轴标签
        # ax[0].legend(loc='upper left')
        # ax[0].set_ylabel('Rewards')
        # ax[0].set_ylim(0, 1)  # 固定reward的纵轴范围为[0, 1]

        # 绘制rewards和value的散点图
        # 使用不同颜色和符号来区分
        ax[0].scatter(plot_timesteps, true_rewards_filtered, color='g', label='True Rewards', marker='o', s=80)
        ax[0].scatter(plot_timesteps, predict_rewards_filtered, color='r', label='Predict Rewards', marker='x',
                      s=80)
        ax[0].set_xticks(plot_timesteps)
        ax[0].set_xticklabels(plot_timesteps)
        ax[0].legend(loc='upper left')
        ax[0].set_ylabel('Rewards')
        ax[0].set_ylim(0, 1)  # 固定reward的纵轴范围为[0, 1]

        # TODO: predict value
        # ax0_twin = ax[0].twinx()
        # ax0_twin.plot(plot_timesteps, predict_value_filtered, 'b--', label='Predict Value')
        # ax0_twin.legend(loc='upper right')
        # ax0_twin.set_ylabel('Value')

        # 绘制原始图像
        image_width = 1.0
        image_height = original_images.shape[3] / original_images.shape[4] * image_width
        gap_width = 0.2
        for i, t in enumerate(plot_timesteps):
            original_image = original_images[batch_idx, t - 1, :, :, :]

            left = i * (image_width + gap_width)
            right = left + image_width
            bottom = 0.5 - image_height / 2
            top = 0.5 + image_height / 2

            ax[1].imshow(torchvision.transforms.ToPILImage()(original_image), extent=[left, right, bottom, top],
                         aspect='auto')
            ax[1].text(left + image_width / 2, bottom - 0.05, f'{t}', ha='center', va='top',
                       fontsize=10)  # 添加时间步索引标签

        ax[1].set_xlim(0, len(plot_timesteps) * (image_width + gap_width) - gap_width)
        ax[1].set_xticks([(i + 0.5) * (image_width + gap_width) for i in range(len(plot_timesteps))])
        ax[1].set_xticklabels([])
        ax[1].set_yticks([])
        ax[1].set_ylabel('Original', rotation=0, labelpad=30)

        # 绘制predict_policy的概率分布柱状图
        bar_width = 0.8 / num_actions
        offset = np.linspace(-0.4 + bar_width / 2, 0.4 - bar_width / 2, num_actions)

        for i, t in enumerate(plot_timesteps):
            for j in range(num_actions):
                ax[2].bar(i + offset[j], predict_policy[batch_idx, t - 1, j].item(), width=bar_width,
                          color=colors[j], alpha=0.5)

        ax[2].set_xticks(range(len(plot_timesteps)))
        ax[2].set_xticklabels(plot_timesteps)
        ax[2].set_ylabel('Predict Policy')
        # 添加图例
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i], alpha=0.5) for i in range(num_actions)]
        labels = [f'Action {i}' for i in range(num_actions)]
        ax[2].legend(handles, labels, loc='upper right', ncol=num_actions)

        # 绘制target_policy的概率分布柱状图
        bar_width = 0.8 / num_actions
        offset = np.linspace(-0.4 + bar_width / 2, 0.4 - bar_width / 2, num_actions)
        for i, t in enumerate(plot_timesteps):
            for j in range(num_actions):
                ax[3].bar(i + offset[j], target_policy[batch_idx, t - 1, j].item(), width=bar_width,
                          color=colors[j], alpha=0.5)
        ax[3].set_xticks(range(len(plot_timesteps)))
        ax[3].set_xticklabels(plot_timesteps)
        ax[3].set_ylabel('MCTS Policy')
        # 添加图例
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i], alpha=0.5) for i in range(num_actions)]
        labels = [f'Action {i}' for i in range(num_actions)]
        ax[3].legend(handles, labels, loc='upper right', ncol=num_actions)

        plt.tight_layout()
        directory = f'/mnt/afs/niuyazhe/code/LightZero/render/{suffix}'
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(f'{directory}/visualization_sequence_{batch_idx}_reward_value_img_policy_mcts-policy.png')
        plt.close()
        # sys.exit(0)  # TODO


def plot_latent_tsne_each_and_all_for_pong(obs_embeddings, suffix='pong'):
    # # 假设 obs_embeddings 是一个 torch.Tensor，形状为 [608, 1, 64]
    # obs_embeddings = torch.randn(608, 1, 768)  # 示例数据，实际使用时替换为实际的 obs_embeddings
    #
    # # 去掉第二个维度（1）
    obs_embeddings = obs_embeddings.squeeze(1)

    # 分割为 8 局，每局 76 个时间步
    num_episodes = 1
    timesteps_per_episode = 10
    episodes = np.split(obs_embeddings.cpu().detach().numpy(), num_episodes)

    # 创建一个颜色列表
    # colors = ['red'] * 1 + ['green'] * 60 + ['blue'] * 15

    # 函数：为每一局绘制 t-SNE 降维结果并保存为 PNG
    def plot_tsne(embeddings_2d, timesteps, title, filename):
        plt.figure(figsize=(10, 8))
        for i, (x, y) in enumerate(embeddings_2d):
            # plt.scatter(x, y, color=colors[i])
            plt.scatter(x, y, color='red')
            plt.text(x, y, str(timesteps[i]), fontsize=9, ha='right')
        plt.title(title)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.grid(True)
        directory = f'/mnt/afs/niuyazhe/code/LightZero/render/{suffix}'
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(f'{directory}/latent_tsne_{filename}')
        # plt.savefig(filename)
        plt.close()

    # 分别处理每一局并保存为 PNG
    for episode_idx, episode in enumerate(episodes):
        # 确认 episode 是一个 numpy 数组或其他适当的数据结构
        n_samples = episode.shape[0]
        # 设置 perplexity 小于 n_samples
        perplexity = min(30, n_samples - 1)  # 选择一个合理的值，比如 30

        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        # tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(episode)

        timesteps = np.arange(timesteps_per_episode)
        plot_tsne(embeddings_2d, timesteps, f'TSNE of Latent States in Episode {episode_idx + 1}',
                  f'episode_{episode_idx + 1}.png')

    # 将所有局的降维结果绘制在一张图中并保存为 PNG
    # plt.figure(figsize=(14, 12))
    # for episode_idx, episode in enumerate(episodes):
    #     tsne = TSNE(n_components=2, random_state=42)
    #     embeddings_2d = tsne.fit_transform(episode)
    #     timesteps = np.arange(timesteps_per_episode) + episode_idx * timesteps_per_episode
    #     for i, (x, y) in enumerate(embeddings_2d):
    #         plt.scatter(x, y, color=colors[i % timesteps_per_episode])
    #         plt.text(x, y, str(timesteps[i]), fontsize=9, ha='right')
    # plt.title('TSNE of Latent States of 8 Episodes')
    # # plt.title('TSNE of Latent States of All Episodes Combined')
    # plt.xlabel('Component 1')
    # plt.ylabel('Component 2')
    # plt.grid(True)

    # directory = f'/mnt/afs/niuyazhe/code/LightZero/render/{suffix}'
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    # plt.savefig(f'{directory}/latent_tsne_all_episodes_combined.png')
    # plt.close()


def plot_latent_tsne_each_and_all_for_visualmatch(obs_embeddings, suffix='visualmatch'):
    # # 假设 obs_embeddings 是一个 torch.Tensor，形状为 [608, 1, 64]
    # obs_embeddings = torch.randn(608, 1, 64)  # 示例数据，实际使用时替换为实际的 obs_embeddings
    #
    # # 去掉第二个维度（1）
    obs_embeddings = obs_embeddings.squeeze(1)

    # 分割为 8 局，每局 76 个时间步
    num_episodes = 8
    timesteps_per_episode = 76
    episodes = np.split(obs_embeddings.cpu().detach().numpy(), num_episodes)

    # 创建一个颜色列表
    colors = ['red'] * 1 + ['green'] * 60 + ['blue'] * 15

    # 函数：为每一局绘制 t-SNE 降维结果并保存为 PNG
    def plot_tsne(embeddings_2d, timesteps, title, filename):
        plt.figure(figsize=(10, 8))
        for i, (x, y) in enumerate(embeddings_2d):
            plt.scatter(x, y, color=colors[i])
            plt.text(x, y, str(timesteps[i]), fontsize=9, ha='right')
        plt.title(title)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.grid(True)
        directory = f'/mnt/afs/niuyazhe/code/LightZero/render/{suffix}'
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(f'{directory}/latent_tsne_{filename}')
        # plt.savefig(filename)
        plt.close()

    # 分别处理每一局并保存为 PNG
    for episode_idx, episode in enumerate(episodes):
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(episode)
        timesteps = np.arange(timesteps_per_episode)
        plot_tsne(embeddings_2d, timesteps, f'TSNE of Latent States in Episode {episode_idx + 1}', f'episode_{episode_idx + 1}.png')

    # 将所有局的降维结果绘制在一张图中并保存为 PNG
    plt.figure(figsize=(14, 12))
    for episode_idx, episode in enumerate(episodes):
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(episode)
        timesteps = np.arange(timesteps_per_episode) + episode_idx * timesteps_per_episode
        for i, (x, y) in enumerate(embeddings_2d):
            plt.scatter(x, y, color=colors[i % timesteps_per_episode])
            plt.text(x, y, str(timesteps[i]), fontsize=9, ha='right')
    plt.title('TSNE of Latent States of 8 Episodes')
    # plt.title('TSNE of Latent States of All Episodes Combined')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True)

    directory = f'/mnt/afs/niuyazhe/code/LightZero/render/{suffix}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(f'{directory}/latent_tsne_all_episodes_combined.png')
    # plt.savefig('all_episodes_combined.png')
    plt.close()


def visualize_reconstruction_v3(original_images, reconstructed_images, target_predict_value, true_rewards,
                                target_policy, predict_value, predict_rewards, predict_policy,
                                not_plot_timesteps=[], suffix='pong', width=64):
    # 确保输入张量的维度匹配
    assert original_images.shape[0] == reconstructed_images.shape[0] // original_images.shape[1]
    assert original_images.shape[1] == reconstructed_images.shape[0] // original_images.shape[0]
    assert original_images.shape[2:] == reconstructed_images.shape[1:]

    batch_size = original_images.shape[0]
    num_timesteps = original_images.shape[1]
    num_actions = predict_policy.shape[2]

    # 根据动作空间大小自适应颜色
    colors = plt.cm.viridis(np.linspace(0, 1, num_actions))

    for batch_idx in range(batch_size):
        fig, ax = plt.subplots(5, 1, figsize=(20, 15), gridspec_kw={'height_ratios': [1, 1, 1, 1, 1]})

        # 绘制rewards和value的折线图
        timesteps = range(1, num_timesteps + 1)
        plot_timesteps = [t for t in timesteps if t not in not_plot_timesteps]

        true_rewards_filtered = [true_rewards[batch_idx, t - 1, 0].cpu().detach().numpy() for t in plot_timesteps]
        predict_rewards_filtered = [predict_rewards[batch_idx, t - 1, 0].cpu().detach().numpy() for t in
                                    plot_timesteps]
        predict_value_filtered = [predict_value[batch_idx, t - 1, 0].cpu().detach().numpy() for t in plot_timesteps]

        ax[0].plot(plot_timesteps, true_rewards_filtered, 'g-', label='True Rewards')
        ax[0].plot(plot_timesteps, predict_rewards_filtered, 'g--', label='Predict Rewards')
        ax[0].set_xticks(plot_timesteps)
        ax[0].set_xticklabels([])
        ax[0].legend(loc='upper left')
        ax[0].set_ylabel('Rewards')
        ax[0].set_ylim(0, 1)  # 固定reward的纵轴范围为[0, 1]

        ax0_twin = ax[0].twinx()
        ax0_twin.plot(plot_timesteps, predict_value_filtered, 'b--', label='Predict Value')
        ax0_twin.legend(loc='upper right')
        ax0_twin.set_ylabel('Value')

        # 绘制原始图像和重建图像
        image_width = 1.0
        image_height = original_images.shape[3] / original_images.shape[4] * image_width
        gap_width = 0.2
        for i in range(num_timesteps):
            if i + 1 not in not_plot_timesteps:
                original_image = original_images[batch_idx, i, :, :, :]
                reconstructed_image = reconstructed_images[i * batch_size + batch_idx, :, :, :]

                left = plot_timesteps.index(i + 1) * (image_width + gap_width)
                right = left + image_width
                bottom = 0.5 - image_height / 2
                top = 0.5 + image_height / 2

                ax[1].imshow(torchvision.transforms.ToPILImage()(original_image), extent=[left, right, bottom, top],
                             aspect='auto')
                ax[2].imshow(torchvision.transforms.ToPILImage()(reconstructed_image),
                             extent=[left, right, bottom, top], aspect='auto')

        ax[1].set_xlim(0, len(plot_timesteps) * (image_width + gap_width) - gap_width)
        ax[1].set_xticks([(i + 0.5) * (image_width + gap_width) for i in range(len(plot_timesteps))])
        ax[1].set_xticklabels([])
        ax[1].set_yticks([])
        ax[1].set_ylabel('Original', rotation=0, labelpad=30)

        ax[2].set_xlim(0, len(plot_timesteps) * (image_width + gap_width) - gap_width)
        ax[2].set_xticks([(i + 0.5) * (image_width + gap_width) for i in range(len(plot_timesteps))])
        ax[2].set_xticklabels([])
        ax[2].set_yticks([])
        ax[2].set_ylabel('Reconstructed', rotation=0, labelpad=30)

        # 绘制predict_policy和target_policy的概率分布柱状图
        bar_width = 0.8 / num_actions
        offset = np.linspace(-0.4 + bar_width / 2, 0.4 - bar_width / 2, num_actions)

        for i in range(num_timesteps):
            if i + 1 not in not_plot_timesteps:
                for j in range(num_actions):
                    ax[3].bar(plot_timesteps.index(i + 1) + offset[j], predict_policy[batch_idx, i, j].item(),
                              width=bar_width, color=colors[j], alpha=0.5)
                    ax[4].bar(plot_timesteps.index(i + 1) + offset[j], target_policy[batch_idx, i, j].item(),
                              width=bar_width, color=colors[j], alpha=0.5)

        ax[3].set_xticks(plot_timesteps)
        ax[3].set_ylabel('Predict Policy')
        ax[4].set_xticks(plot_timesteps)
        ax[4].set_xlabel('Timestep')
        ax[4].set_ylabel('Target Policy')

        # 添加图例
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i], alpha=0.5) for i in range(num_actions)]
        labels = [f'Action {i}' for i in range(num_actions)]
        ax[4].legend(handles, labels, loc='upper right', ncol=num_actions)

        plt.tight_layout()
        directory = f'/mnt/afs/niuyazhe/code/LightZero/render/{suffix}'
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(f'{directory}/reconstruction_visualization_batch_{batch_idx}_v3.png')
        plt.close()


def visualize_reconstruction_v2(original_images, reconstructed_images, target_predict_value, true_rewards,
                                target_policy, predict_value, predict_rewards, predict_policy, suffix='pong',
                                width=64):
    # 确保输入张量的维度匹配
    assert original_images.shape[0] == reconstructed_images.shape[0] // original_images.shape[1]
    assert original_images.shape[1] == reconstructed_images.shape[0] // original_images.shape[0]
    assert original_images.shape[2:] == reconstructed_images.shape[1:]

    batch_size = original_images.shape[0]
    num_timesteps = original_images.shape[1]
    num_actions = predict_policy.shape[2]

    # 根据动作空间大小自适应颜色
    colors = plt.cm.viridis(np.linspace(0, 1, num_actions))
    # colors = ['r', 'g', 'b', 'y']

    for batch_idx in range(batch_size):
        fig, ax = plt.subplots(5, 1, figsize=(20, 15), gridspec_kw={'height_ratios': [1, 1, 1, 1, 1]})

        # 绘制rewards和value的折线图
        timesteps = range(1, num_timesteps + 1)
        ax[0].plot(timesteps, true_rewards[batch_idx, :, 0].cpu().detach().numpy(), 'g-', label='True Rewards')
        ax[0].plot(timesteps, predict_rewards[batch_idx, :, 0].cpu().detach().numpy(), 'g--',
                   label='Predict Rewards')
        ax[0].set_xticks(timesteps)
        ax[0].set_xticklabels([])
        ax[0].legend(loc='upper left')
        ax[0].set_ylabel('Rewards')

        ax0_twin = ax[0].twinx()
        # ax0_twin.plot(timesteps, target_predict_value[batch_idx, :, 0].cpu().detach().numpy(), 'b-', label='Target Predict Value')
        ax0_twin.plot(timesteps, predict_value[batch_idx, :, 0].cpu().detach().numpy(), 'b--',
                      label='Predict Value')
        ax0_twin.legend(loc='upper right')
        ax0_twin.set_ylabel('Value')

        # 绘制原始图像和重建图像
        image_width = 1.0
        image_height = original_images.shape[3] / original_images.shape[4] * image_width
        gap_width = 0.2
        for i in range(num_timesteps):
            original_image = original_images[batch_idx, i, :, :, :]
            reconstructed_image = reconstructed_images[i * batch_size + batch_idx, :, :, :]

            left = i * (image_width + gap_width)
            right = left + image_width
            bottom = 0.5 - image_height / 2
            top = 0.5 + image_height / 2

            ax[1].imshow(torchvision.transforms.ToPILImage()(original_image), extent=[left, right, bottom, top],
                         aspect='auto')
            ax[2].imshow(torchvision.transforms.ToPILImage()(reconstructed_image),
                         extent=[left, right, bottom, top], aspect='auto')

        ax[1].set_xlim(0, num_timesteps * (image_width + gap_width) - gap_width)
        ax[1].set_xticks([(i + 0.5) * (image_width + gap_width) for i in range(num_timesteps)])
        ax[1].set_xticklabels([])
        ax[1].set_yticks([])
        ax[1].set_ylabel('Original', rotation=0, labelpad=30)

        ax[2].set_xlim(0, num_timesteps * (image_width + gap_width) - gap_width)
        ax[2].set_xticks([(i + 0.5) * (image_width + gap_width) for i in range(num_timesteps)])
        ax[2].set_xticklabels([])
        ax[2].set_yticks([])
        ax[2].set_ylabel('Reconstructed', rotation=0, labelpad=30)

        # # 绘制predict_policy和target_policy的概率分布柱状图
        # 计算柱状图的宽度和偏移量，确保它们不会重叠
        bar_width = 0.8 / num_actions
        offset = np.linspace(-0.4 + bar_width / 2, 0.4 - bar_width / 2, num_actions)
        # 绘制predict_policy和target_policy的概率分布柱状图
        for i in range(num_timesteps):
            for j in range(num_actions):
                ax[3].bar(i + offset[j], predict_policy[batch_idx, i, j].item(), width=bar_width, color=colors[j],
                          alpha=0.5)
                ax[4].bar(i + offset[j], target_policy[batch_idx, i, j].item(), width=bar_width, color=colors[j],
                          alpha=0.5)
        ax[3].set_xticks(timesteps)
        ax[3].set_ylabel('Predict Policy')
        ax[4].set_xticks(timesteps)
        ax[4].set_xlabel('Timestep')
        ax[4].set_ylabel('Target Policy')
        # 添加图例
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i], alpha=0.5) for i in range(num_actions)]
        labels = [f'Action {i}' for i in range(num_actions)]
        ax[4].legend(handles, labels, loc='upper right', ncol=num_actions)

        plt.tight_layout()
        directory = f'/mnt/afs/niuyazhe/code/LightZero/render/{suffix}'
        # 检查路径是否存在，不存在则创建
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(f'{directory}/reconstruction_visualization_batch_{batch_idx}_v2.png')
        # plt.savefig(f'./render/{suffix}/reconstruction_visualization_batch_{batch_idx}_v2.png')
        plt.close()

def visualize_reconstruction_v1(original_images, reconstructed_images, suffix='pong', width=64):
    # 确保输入张量的维度匹配
    assert original_images.shape[0] == reconstructed_images.shape[0] // original_images.shape[1]
    assert original_images.shape[1] == reconstructed_images.shape[0] // original_images.shape[0]
    assert original_images.shape[2:] == reconstructed_images.shape[1:]

    batch_size = original_images.shape[0]
    num_timesteps = original_images.shape[1]

    for batch_idx in range(batch_size):
        # 创建一个白色背景的大图像
        big_image = torch.ones(3, (width + 1) * 2 + 1, (width + 1) * num_timesteps + 1)

        # 将原始图像和重建图像复制到大图像中
        for i in range(num_timesteps):
            original_image = original_images[batch_idx, i, :, :, :]
            reconstructed_image = reconstructed_images[i * batch_size + batch_idx, :, :, :]

            big_image[:, 1:1 + width, (width + 1) * i + 1:(width + 1) * (i + 1)] = original_image
            big_image[:, 2 + width:2 + 2 * width, (width + 1) * i + 1:(width + 1) * (i + 1)] = reconstructed_image

        # 转换张量为PIL图像
        image = torchvision.transforms.ToPILImage()(big_image)

        # 绘制图像
        plt.figure(figsize=(20, 4))
        plt.imshow(image)
        plt.axis('off')

        # 添加时间步标签
        for i in range(num_timesteps):
            plt.text((width + 1) * i + width / 2, -10, str(i + 1), ha='center', va='top', fontsize=12)

        # 添加行标签
        plt.text(-0.5, 3, 'Original', ha='right', va='center', fontsize=12)
        plt.text(-0.5, 3 + width + 1, 'Reconstructed', ha='right', va='center', fontsize=12)

        plt.tight_layout()
        # plt.savefig(f'./render/{suffix}/reconstruction_visualization_batch_{batch_idx}_v1.png')
        plt.savefig(
            f'/mnt/afs/niuyazhe/code/LightZero/render/{suffix}/reconstruction_visualization_batch_{batch_idx}_v1.png')
        plt.close()


def save_as_image_with_timestep(batch_tensor, suffix='pong'):
    # batch_tensor 的形状应该是 [batch_size, sequence_length, channels, height, width]
    # 在这里 channels = 4, height = 5, width = 5
    batch_size, sequence_length, channels, height, width = batch_tensor.shape

    # 为了将所有帧组合成一张图，我们设置每行显示 sequence_length 个图像
    rows = batch_size
    cols = sequence_length

    # 创建一个足够大的空白图像来容纳所有的帧
    # 每个RGB图像的大小是 height x width，总图像的大小是 (rows * height + text_height) x (cols * width)
    text_height = 20  # 留出空间用于显示时间步索引
    total_height = rows * (height + text_height)
    final_image = Image.new('RGB', (cols * width, total_height), color='white')

    # 遍历每一帧，将其转换为PIL图像，并粘贴到正确的位置
    for i in range(rows):
        for j in range(cols):
            # 提取当前帧的前三个通道（假设前三个通道是RGB）
            frame = batch_tensor[i, j, :3, :, :]
            # 转换为numpy数组，并调整数据范围为0-255
            frame = frame.mul(255).byte().cpu().detach().numpy().transpose(1, 2, 0)
            # 创建一个PIL图像
            img = Image.fromarray(frame)
            # 粘贴到最终图像的相应位置
            final_image.paste(img, (j * width, i * (height + text_height)))

            # 在图像下方添加时间步索引
            draw = ImageDraw.Draw(final_image)
            # text = f"Timestep {j}"
            text = f"{j}"
            # 使用默认字体
            draw.text((j * width, i * (height + text_height) + height), text, fill="black")

    # 保存图像
    directory = f'/mnt/afs/niuyazhe/code/LightZero/render/{suffix}'
    # 检查路径是否存在，不存在则创建
    if not os.path.exists(directory):
        os.makedirs(directory)
    final_image.save(f'{directory}/visualization_batch_with_timestep.png')


def save_as_image(batch_tensor, suffix='pong'):
    # # 假设 batch['observations'] 是一个满足条件的tensor
    # # 示例tensor，实际使用中应替换为实际的tensor数据
    # batch = {'observations': torch.randn(3, 16, 4, 5, 5)}

    # # 调用函数
    # save_as_image(batch['observations'])

    # batch_tensor 的形状应该是 [batch_size, sequence_length, channels, height, width]
    # 在这里 channels = 4, height = 5, width = 5
    batch_size, sequence_length, channels, height, width = batch_tensor.shape

    # 为了将所有帧组合成一张图，我们设置每行显示 sequence_length 个图像
    rows = batch_size
    cols = sequence_length

    # 创建一个足够大的空白图像来容纳所有的帧
    # 每个RGB图像的大小是 height x width，总图像的大小是 (rows * height) x (cols * width)
    final_image = Image.new('RGB', (cols * width, rows * height))

    # 遍历每一帧，将其转换为PIL图像，并粘贴到正确的位置
    for i in range(rows):
        for j in range(cols):
            # 提取当前帧的前三个通道（假设前三个通道是RGB）
            frame = batch_tensor[i, j, :3, :, :]
            # 转换为numpy数组，并调整数据范围为0-255
            frame = frame.mul(255).byte().cpu().detach().numpy().transpose(1, 2, 0)
            # 创建一个PIL图像
            img = Image.fromarray(frame)
            # 粘贴到最终图像的相应位置
            final_image.paste(img, (j * width, i * height))

    # 保存图像
    directory = f'/mnt/afs/niuyazhe/code/LightZero/render/{suffix}'
    # 检查路径是否存在，不存在则创建
    if not os.path.exists(directory):
        os.makedirs(directory)
    final_image.save(f'{directory}/visualization_batch.png')
    plt.close()




def render_img(obs: int, rec_img: int):
    from PIL import Image
    import matplotlib.pyplot as plt

    # 假设batch是一个字典,其中包含了observations键,
    # 并且它的形状是torch.Size([B, N, C, H, W])
    # batch_observations = batch_for_gpt['observations']
    # batch_observations = batch['observations']
    batch_observations = obs.unsqueeze(0)
    # batch_observations = rec_img.unsqueeze(0)

    # batch_observations = observations.unsqueeze(0)
    # batch_observations = x.unsqueeze(0)
    # batch_observations = reconstructions.unsqueeze(0)

    B, N, C, H, W = batch_observations.shape  # 自动检测维度

    # 分隔条的宽度(可以根据需要调整)
    separator_width = 2

    # 遍历每个样本
    for i in range(B):
        # 提取当前样本中的所有帧
        frames = batch_observations[i]

        # 计算拼接图像的总宽度(包括分隔条)
        total_width = N * W + (N - 1) * separator_width

        # 创建一个新的图像,其中包含分隔条
        concat_image = Image.new('RGB', (total_width, H), color='black')

        # 拼接每一帧及分隔条
        for j in range(N):
            frame = frames[j].permute(1, 2, 0).cpu().detach().numpy()  # 转换为(H, W, C)
            frame_image = Image.fromarray((frame * 255).astype('uint8'), 'RGB')

            # 计算当前帧在拼接图像中的位置
            x_position = j * (W + separator_width)
            concat_image.paste(frame_image, (x_position, 0))

        # 显示图像
        plt.imshow(concat_image)
        plt.title(f'Sample {i + 1}')
        plt.axis('off')  # 关闭坐标轴显示
        plt.show()

        # 保存图像到文件
        concat_image.save(f'render/sample_{i + 1}.png')
