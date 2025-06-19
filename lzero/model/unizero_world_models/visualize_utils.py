import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image
from PIL import ImageDraw
from sklearn.manifold import TSNE


def visualize_sequence_only(
    original_images,
    not_plot_timesteps=None,
):
    """
    Plots only the original frames for each sequence in a batch.

    Args:
        original_images (torch.Tensor or ndarray):
            Shape (batch_size, T, C, H, W) or (T, C, H, W).
        not_plot_timesteps (list[int], optional):
            1-based timesteps to skip. Default None.
    """
    if not_plot_timesteps is None:
        not_plot_timesteps = []

    orig = (original_images.detach().cpu().numpy()
            if hasattr(original_images, 'detach') else original_images)

    if orig.ndim == 5:
        batch_size, T, C, H, W = orig.shape
    elif orig.ndim == 4:
        orig = orig[None, ...]
        batch_size, T, C, H, W = orig.shape
    else:
        raise ValueError(f"Expected 4D or 5D input, got shape {orig.shape}")

    # Fixed plotting parameters
    frame_width = 1.0
    gap_width = 0.2
    frame_height = (H / W) * frame_width

    for b in range(batch_size):
        seq = orig[b]  # shape (T, C, H, W)
        all_ts = list(range(1, T + 1))
        plot_ts = [t for t in all_ts if t not in not_plot_timesteps]
        N = len(plot_ts)

        fig, axes = plt.subplots(
            1, N,
            figsize=(frame_width * N, frame_width * frame_height),
            squeeze=False
        )

        for i, t in enumerate(plot_ts):
            left   = i * (frame_width + gap_width)
            right  = left + frame_width
            bottom = 0.5 - frame_height / 2
            top    = 0.5 + frame_height / 2

            img = seq[t - 1].transpose(1, 2, 0).astype('uint8')
            ax  = axes[0, i]
            ax.imshow(img, extent=[left, right, bottom, top], aspect='auto')
            ax.text(left + frame_width/2, bottom - 0.05,
                    str(t), ha='center', va='top', fontsize=10)
            ax.axis('off')

        plt.tight_layout()

    filename = "frames.pdf" # Change to other filename if needed
    plt.savefig(filename)
    plt.close(fig)
    print(f"Saved {filename}")

def visualize_reward_value_img_policy( original_images, reconstructed_images, target_predict_value, true_rewards,
                                      target_policy, predict_value, predict_rewards, predict_policy,
                                      not_plot_timesteps=[], suffix='pong', width=64
                                      ):
    """
    Visualizes the rewards, values, original images, and policy distributions over time for a batch of sequences.

    Arguments:
        - original_images (:obj:`torch.Tensor`): The original input images with shape (batch_size, num_timesteps, channels, height, width).
        - reconstructed_images (:obj:`torch.Tensor`): The reconstructed images with shape (batch_size * num_timesteps, channels, height, width).
        - target_predict_value (:obj:`torch.Tensor`): The target predicted values.
        - true_rewards (:obj:`torch.Tensor`): The true rewards with shape (batch_size, num_timesteps, 1).
        - target_policy (:obj:`torch.Tensor`): The target policy distribution with shape (batch_size, num_timesteps, num_actions).
        - predict_value (:obj:`torch.Tensor`): The predicted values with shape (batch_size, num_timesteps, 1).
        - predict_rewards (:obj:`torch.Tensor`): The predicted rewards with shape (batch_size, num_timesteps, 1).
        - predict_policy (:obj:`torch.Tensor`): The predicted policy distribution with shape (batch_size, num_timesteps, num_actions).
        - not_plot_timesteps (:obj:`list, optional`): A list of timesteps to exclude from plotting. Default is an empty list.
        - suffix (:obj:`str, optional`): A suffix for the output directory. Default is 'pong'.
        - width (:obj:`int, optional`): The width of the images. Default is 64.
    Returns:
        - None
    """
    # Ensure the dimensions of input tensors match
    assert original_images.shape[0] == reconstructed_images.shape[0] // original_images.shape[1]
    assert original_images.shape[1] == reconstructed_images.shape[0] // original_images.shape[0]
    assert original_images.shape[2:] == reconstructed_images.shape[1:]

    batch_size = original_images.shape[0]
    num_timesteps = original_images.shape[1]
    num_actions = predict_policy.shape[2]

    # Adapt colors based on the size of the action space
    colors = plt.cm.viridis(np.linspace(0, 1, num_actions))

    for batch_idx in range(batch_size):
        # Create subplots for rewards, values, original images, and policy distributions
        fig, ax = plt.subplots(4, 1, figsize=(20, 12), gridspec_kw={'height_ratios': [1, 1, 1, 1]})

        # Plot rewards and values as scatter plots
        timesteps = range(1, num_timesteps + 1)
        plot_timesteps = [t for t in timesteps if t not in not_plot_timesteps]

        true_rewards_filtered = [true_rewards[batch_idx, t - 1, 0].cpu().detach().numpy() for t in plot_timesteps]
        predict_rewards_filtered = [predict_rewards[batch_idx, t - 1, 0].cpu().detach().numpy() for t in
                                    plot_timesteps]
        predict_value_filtered = [predict_value[batch_idx, t - 1, 0].cpu().detach().numpy() for t in plot_timesteps]

        ax[0].scatter(plot_timesteps, true_rewards_filtered, color='g', label='True Rewards', marker='o', s=80)
        ax[0].scatter(plot_timesteps, predict_rewards_filtered, color='r', label='Predict Rewards', marker='x',
                      s=80)
        ax[0].set_xticks(plot_timesteps)
        ax[0].set_xticklabels(plot_timesteps)
        ax[0].legend(loc='upper left')
        ax[0].set_ylabel('Rewards')
        ax[0].set_ylim(0, 1)  # Fixed y-axis range for rewards

        # TODO: predict value
        # ax0_twin = ax[0].twinx()
        # ax0_twin.plot(plot_timesteps, predict_value_filtered, 'b--', label='Predict Value')
        # ax0_twin.legend(loc='upper right')
        # ax0_twin.set_ylabel('Value')

        # Plot original images
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
            ax[1].text(left + image_width / 2, bottom - 0.05, f'{t}', ha='center', va='top', fontsize=10)  # Add time step index label

        ax[1].set_xlim(0, len(plot_timesteps) * (image_width + gap_width) - gap_width)
        ax[1].set_xticks([(i + 0.5) * (image_width + gap_width) for i in range(len(plot_timesteps))])
        ax[1].set_xticklabels([])
        ax[1].set_yticks([])
        ax[1].set_ylabel('Original', rotation=0, labelpad=30)

        # Plot predicted policy distributions as bar charts
        bar_width = 0.8 / num_actions
        offset = np.linspace(-0.4 + bar_width / 2, 0.4 - bar_width / 2, num_actions)

        for i, t in enumerate(plot_timesteps):
            for j in range(num_actions):
                ax[2].bar(i + offset[j], predict_policy[batch_idx, t - 1, j].item(), width=bar_width,
                          color=colors[j], alpha=0.5)

        ax[2].set_xticks(range(len(plot_timesteps)))
        ax[2].set_xticklabels(plot_timesteps)
        ax[2].set_ylabel('Predict Policy')
        # Add legend for predicted policy
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i], alpha=0.5) for i in range(num_actions)]
        labels = [f'Action {i}' for i in range(num_actions)]
        ax[2].legend(handles, labels, loc='upper right', ncol=num_actions)

        # Plot target policy distributions as bar charts
        bar_width = 0.8 / num_actions
        offset = np.linspace(-0.4 + bar_width / 2, 0.4 - bar_width / 2, num_actions)
        for i, t in enumerate(plot_timesteps):
            for j in range(num_actions):
                ax[3].bar(i + offset[j], target_policy[batch_idx, t - 1, j].item(), width=bar_width,
                          color=colors[j], alpha=0.5)
        ax[3].set_xticks(range(len(plot_timesteps)))
        ax[3].set_xticklabels(plot_timesteps)
        ax[3].set_ylabel('MCTS Policy')
        # Add legend for target policy
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i], alpha=0.5) for i in range(num_actions)]
        labels = [f'Action {i}' for i in range(num_actions)]
        ax[3].legend(handles, labels, loc='upper right', ncol=num_actions)

        plt.tight_layout()
        directory = f'/your_path/code/LightZero/render_{suffix}'
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(f'{directory}/visualization_sequence_{batch_idx}_reward_value_img_policy_mcts-policy.png')
        print(f'Saved visualization to {directory}/visualization_sequence_{batch_idx}_reward_value_img_policy_mcts-policy.png')
        plt.close()


def plot_latent_tsne_each_and_all_for_pong(obs_embeddings, suffix='pong'):
    """
    Plots t-SNE dimensionality reduction of latent state embeddings for individual episodes and
    saves the plots as PNG files.

    Arguments:
        - obs_embeddings (:obj:`torch.Tensor`): The latent state embeddings with shape (num_samples, 1, embedding_dim).
        - suffix (:obj:`str`): The suffix for the directory name where plots are saved. Default is 'pong'.
    """
    # Remove the second dimension (1)
    obs_embeddings = obs_embeddings.squeeze(1)

    # Split into 8 episodes, each with 76 timesteps
    num_episodes = 1
    timesteps_per_episode = 10
    episodes = np.split(obs_embeddings.cpu().detach().numpy(), num_episodes)

    # Create a list of colors
    # colors = ['red'] * 1 + ['green'] * 60 + ['blue'] * 15

    def plot_tsne(embeddings_2d, timesteps, title, filename):
        """
        Plots the 2D t-SNE embeddings and saves the plot as a PNG file.

        Arguments:
            - embeddings_2d (np.ndarray): The 2D t-SNE embeddings.
            - timesteps (np.ndarray): The timesteps corresponding to the embeddings.
            - title (:obj:`str`): The title of the plot.
            - filename (:obj:`str`): The filename for saving the plot.
        Returns:
            - None
        """
        plt.figure(figsize=(10, 8))
        for i, (x, y) in enumerate(embeddings_2d):
            # plt.scatter(x, y, color=colors[i])
            plt.scatter(x, y, color='red')
            plt.text(x, y, str(timesteps[i]), fontsize=9, ha='right')
        plt.title(title)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.grid(True)
        directory = f'/your_path/code/LightZero/render/{suffix}'
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(f'{directory}/latent_tsne_{filename}')
        # plt.savefig(filename)
        plt.close()

    # Process and save each episode as a PNG
    for episode_idx, episode in enumerate(episodes):
        # Ensure episode is a numpy array or an appropriate data structure
        n_samples = episode.shape[0]
        # Set perplexity less than n_samples
        perplexity = min(30, n_samples - 1)  # Choose a reasonable value, such as 30

        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        embeddings_2d = tsne.fit_transform(episode)

        timesteps = np.arange(timesteps_per_episode)
        plot_tsne(embeddings_2d, timesteps, f'TSNE of Latent States in Episode {episode_idx + 1}',
                  f'episode_{episode_idx + 1}.png')


def plot_latent_tsne_each_and_all_for_visualmatch(obs_embeddings, suffix='visualmatch'):
    """
    This function visualizes the t-SNE dimensionality reduction results of latent state embeddings.
    It processes the embeddings for multiple episodes and generates both individual plots for each episode
    and a combined plot for all episodes.

    Arguments:
        - obs_embeddings (Tensor): The observations embeddings tensor of shape (num_episodes * timesteps_per_episode, 1, embedding_dim).
        - suffix (:obj:`str`): The suffix for the output directory where the plots will be saved.

    The function performs the following steps:
        1. Removes the second dimension (1) from `obs_embeddings`.
        2. Splits the embeddings into 8 episodes, each containing 76 timesteps.
        3. Creates a list of colors for plotting.
        4. Defines a nested function `plot_tsne` to plot and save t-SNE results.
        5. Processes each episode to plot and save the t-SNE results individually.
        6. Plots and saves the t-SNE results for all episodes combined in a single plot.
    """
    # Remove the second dimension (1)
    obs_embeddings = obs_embeddings.squeeze(1)

    # Split into 8 episodes, each with 76 timesteps
    num_episodes = 8
    timesteps_per_episode = 76
    episodes = np.split(obs_embeddings.cpu().detach().numpy(), num_episodes)

    # Create a list of colors
    colors = ['red'] * 1 + ['green'] * 60 + ['blue'] * 15

    # Function to plot t-SNE results for each episode and save as PNG
    def plot_tsne(embeddings_2d, timesteps, title, filename):
        plt.figure(figsize=(10, 8))
        for i, (x, y) in enumerate(embeddings_2d):
            plt.scatter(x, y, color=colors[i])
            plt.text(x, y, str(timesteps[i]), fontsize=9, ha='right')
        plt.title(title)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.grid(True)
        directory = f'/your_path/code/LightZero/render/{suffix}'
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(f'{directory}/latent_tsne_{filename}')
        plt.close()

    # Process each episode separately and save as PNG
    for episode_idx, episode in enumerate(episodes):
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(episode)
        timesteps = np.arange(timesteps_per_episode)
        plot_tsne(embeddings_2d, timesteps, f'TSNE of Latent States in Episode {episode_idx + 1}', f'episode_{episode_idx + 1}.png')

    # Plot and save the combined t-SNE results of all episodes
    plt.figure(figsize=(14, 12))
    for episode_idx, episode in enumerate(episodes):
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(episode)
        timesteps = np.arange(timesteps_per_episode) + episode_idx * timesteps_per_episode
        for i, (x, y) in enumerate(embeddings_2d):
            plt.scatter(x, y, color=colors[i % timesteps_per_episode])
            plt.text(x, y, str(timesteps[i]), fontsize=9, ha='right')
    plt.title('TSNE of Latent States of 8 Episodes')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True)

    directory = f'/your_path/code/LightZero/render/{suffix}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(f'{directory}/latent_tsne_all_episodes_combined.png')
    plt.close()


def visualize_reconstruction_v2(original_images, reconstructed_images, target_predict_value, true_rewards,
                                target_policy, predict_value, predict_rewards, predict_policy,
                                not_plot_timesteps=[], suffix='pong', width=64):
    """
    Visualizes the reconstruction of a sequence of images and the performance of different policies.

    Arguments:
        - original_images (:obj:`torch.Tensor`): Tensor of original images of shape (batch_size, num_timesteps, channels, height, width).
        - reconstructed_images (:obj:`torch.Tensor`): Tensor of reconstructed images of shape (batch_size * num_timesteps, channels, height, width).
        - target_predict_value (:obj:`torch.Tensor`): Tensor of target predicted values.
        - true_rewards (:obj:`torch.Tensor`): Tensor of true rewards of shape (batch_size, num_timesteps, 1).
        - target_policy (:obj:`torch.Tensor`): Tensor of target policy probabilities of shape (batch_size, num_timesteps, num_actions).
        - predict_value (:obj:`torch.Tensor`): Tensor of predicted values of shape (batch_size, num_timesteps, 1).
        - predict_rewards (:obj:`torch.Tensor`): Tensor of predicted rewards of shape (batch_size, num_timesteps, 1).
        - predict_policy (:obj:`torch.Tensor`): Tensor of predicted policy probabilities of shape (batch_size, num_timesteps, num_actions).
        - not_plot_timesteps (list, optional): List of timesteps to exclude from the plot. Defaults to [].
        - suffix (str, optional): Suffix for the output file name. Defaults to 'pong'.
        - width (int, optional): Width of the images in the plot. Defaults to 64.

    Notes:
        - The function saves the visualizations as PNG files in the specified directory.
        - It plots the true and predicted rewards, values, original and reconstructed images, and policy probabilities.
    """
    # Ensure the dimensions of the input tensors match
    assert original_images.shape[0] == reconstructed_images.shape[0] // original_images.shape[1]
    assert original_images.shape[1] == reconstructed_images.shape[0] // original_images.shape[0]
    assert original_images.shape[2:] == reconstructed_images.shape[1:]

    batch_size = original_images.shape[0]
    num_timesteps = original_images.shape[1]
    num_actions = predict_policy.shape[2]

    # Adapt colors based on the number of actions
    colors = plt.cm.viridis(np.linspace(0, 1, num_actions))

    for batch_idx in range(batch_size):
        fig, ax = plt.subplots(5, 1, figsize=(20, 15), gridspec_kw={'height_ratios': [1, 1, 1, 1, 1]})

        # Plot the rewards and values as line charts
        timesteps = range(1, num_timesteps + 1)
        plot_timesteps = [t for t in timesteps if t not in not_plot_timesteps]

        true_rewards_filtered = [true_rewards[batch_idx, t - 1, 0].cpu().detach().numpy() for t in plot_timesteps]
        predict_rewards_filtered = [predict_rewards[batch_idx, t - 1, 0].cpu().detach().numpy() for t in plot_timesteps]
        predict_value_filtered = [predict_value[batch_idx, t - 1, 0].cpu().detach().numpy() for t in plot_timesteps]

        ax[0].plot(plot_timesteps, true_rewards_filtered, 'g-', label='True Rewards')
        ax[0].plot(plot_timesteps, predict_rewards_filtered, 'g--', label='Predict Rewards')
        ax[0].set_xticks(plot_timesteps)
        ax[0].set_xticklabels([])
        ax[0].legend(loc='upper left')
        ax[0].set_ylabel('Rewards')
        ax[0].set_ylim(0, 1)  # Fixed y-axis range for rewards

        ax0_twin = ax[0].twinx()
        ax0_twin.plot(plot_timesteps, predict_value_filtered, 'b--', label='Predict Value')
        ax0_twin.legend(loc='upper right')
        ax0_twin.set_ylabel('Value')

        # Plot the original and reconstructed images
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
                             extent=[left, right, bottom, top],
                             aspect='auto')

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

        # Plot the predicted and target policy probability distributions as bar charts
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

        # Add legend for the action colors
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i], alpha=0.5) for i in range(num_actions)]
        labels = [f'Action {i}' for i in range(num_actions)]
        ax[4].legend(handles, labels, loc='upper right', ncol=num_actions)

        plt.tight_layout()
        directory = f'/your_path/code/LightZero/render/{suffix}'
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(f'{directory}/reconstruction_visualization_batch_{batch_idx}_v2.png')
        plt.close()


def visualize_reconstruction_v1(original_images, reconstructed_images, suffix='pong', width=64):
    """
    Visualizes the reconstruction of images by comparing original and reconstructed images side by side.

    Arguments:
        original_images (:obj:`torch.Tensor`): A tensor of original images with shape (batch_size, num_timesteps, channels, height, width).
        reconstructed_images (:obj:`torch.Tensor`): A tensor of reconstructed images with shape (batch_size * num_timesteps, channels, height, width).
        suffix (str, optional): A suffix for the saved image filenames. Default is 'pong'.
        width (int, optional): The width of each image. Default is 64.

    Raises:
        AssertionError: If the dimensions of the input tensors do not match the expected shapes.

    """
    # Ensure the input tensor dimensions match
    assert original_images.shape[0] == reconstructed_images.shape[0] // original_images.shape[1]
    assert original_images.shape[1] == reconstructed_images.shape[0] // original_images.shape[0]
    assert original_images.shape[2:] == reconstructed_images.shape[1:]

    batch_size = original_images.shape[0]
    num_timesteps = original_images.shape[1]

    for batch_idx in range(batch_size):
        # Create a large image with a white background
        big_image = torch.ones(3, (width + 1) * 2 + 1, (width + 1) * num_timesteps + 1)

        # Copy the original and reconstructed images into the large image
        for i in range(num_timesteps):
            original_image = original_images[batch_idx, i, :, :, :]
            reconstructed_image = reconstructed_images[i * batch_size + batch_idx, :, :, :]

            big_image[:, 1:1 + width, (width + 1) * i + 1:(width + 1) * (i + 1)] = original_image
            big_image[:, 2 + width:2 + 2 * width, (width + 1) * i + 1:(width + 1) * (i + 1)] = reconstructed_image

        # Convert the tensor to a PIL image
        image = torchvision.transforms.ToPILImage()(big_image)

        # Plot the image
        plt.figure(figsize=(20, 4))
        plt.imshow(image)
        plt.axis('off')

        # Add timestep labels
        for i in range(num_timesteps):
            plt.text((width + 1) * i + width / 2, -10, str(i + 1), ha='center', va='top', fontsize=12)

        # Add row labels
        plt.text(-0.5, 3, 'Original', ha='right', va='center', fontsize=12)
        plt.text(-0.5, 3 + width + 1, 'Reconstructed', ha='right', va='center', fontsize=12)

        plt.tight_layout()
        plt.savefig(
            f'/your_path/code/LightZero/render/{suffix}/reconstruction_visualization_batch_{batch_idx}_v1.png')
        plt.close()


def save_as_image_with_timestep(batch_tensor, suffix='pong'):
    """
    Saves a batch of image sequences as a single image with timestep annotations.

    Arguments:
        batch_tensor (:obj:`torch.Tensor`): A tensor of shape [batch_size, sequence_length, channels, height, width].
                                     Here, channels = 4, height = 5, width = 5.
        suffix (:obj:`str`): A suffix for the directory where the image will be saved. Default is 'pong'.

    The function will arrange the frames in a grid where each row corresponds to a sequence in the batch
    and each column corresponds to a timestep within that sequence. Each frame is converted to a PIL image,
    and the timestep index is added below each frame.
    """
    # Extract dimensions from the batch tensor
    batch_size, sequence_length, channels, height, width = batch_tensor.shape

    # Set the number of rows and columns for the output image
    rows = batch_size
    cols = sequence_length

    # Create a blank image large enough to hold all frames
    text_height = 20  # Extra space for displaying timestep index
    total_height = rows * (height + text_height)
    final_image = Image.new('RGB', (cols * width, total_height), color='white')

    # Iterate over each frame, convert to PIL image, and paste it in the correct location
    for i in range(rows):
        for j in range(cols):
            # Extract the first three channels (assuming they are RGB)
            frame = batch_tensor[i, j, :3, :, :]
            # Convert to numpy array and scale data to range 0-255
            frame = frame.mul(255).byte().cpu().detach().numpy().transpose(1, 2, 0)
            # Create a PIL image
            img = Image.fromarray(frame)
            # Paste the image at the correct location in the final image
            final_image.paste(img, (j * width, i * (height + text_height)))

            # Add timestep index below the image
            draw = ImageDraw.Draw(final_image)
            text = f"{j}"
            draw.text((j * width, i * (height + text_height) + height), text, fill="black")

    # Define the directory to save the image
    directory = f'/your_path/code/LightZero/render/{suffix}'
    # Check if the directory exists, create if it does not
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Save the final image
    final_image.save(f'{directory}/visualization_batch_with_timestep.png')


def save_as_image(batch_tensor, suffix='pong'):
    """
    Save a batch of image sequences as a single image file.

    This function takes a 5D tensor representing a batch of image sequences and
    combines them into a single image, which is then saved to disk. The resulting
    image will have each sequence arranged in a row, with individual frames
    displayed in columns.

    Arguments:
    - batch_tensor (:obj:`torch.Tensor`): A 5D tensor of shape [batch_size, sequence_length, channels, height, width],
                                   where channels should be at least 4.
    - suffix (:obj:`str`): A suffix to include in the directory path where the image will be saved.

    Example:
    ```python
    # Example tensor, replace with actual tensor data in practice
    batch = {'observations': torch.randn(3, 16, 4, 5, 5)}

    # Call the function
    save_as_image(batch['observations'])
    ```

    Note:
    The function assumes that the first three channels of each frame represent RGB data.

    """

    # batch_tensor should have the shape [batch_size, sequence_length, channels, height, width]
    # Here, channels = 4, height = 5, width = 5
    batch_size, sequence_length, channels, height, width = batch_tensor.shape

    # To combine all frames into one image, we set each row to display sequence_length images
    rows = batch_size
    cols = sequence_length

    # Create a blank image large enough to hold all frames
    # Each RGB image has size height x width, so the total image size is (rows * height) x (cols * width)
    final_image = Image.new('RGB', (cols * width, rows * height))

    # Iterate over each frame, convert it to a PIL image, and paste it in the correct position
    for i in range(rows):
        for j in range(cols):
            # Extract the first three channels of the current frame (assuming the first three channels are RGB)
            frame = batch_tensor[i, j, :3, :, :]
            # Convert to numpy array and adjust data range to 0-255
            frame = frame.mul(255).byte().cpu().detach().numpy().transpose(1, 2, 0)
            # Create a PIL image
            img = Image.fromarray(frame)
            # Paste it at the corresponding position in the final image
            final_image.paste(img, (j * width, i * height))

    # Save the image
    directory = f'/your_path/code/LightZero/render/{suffix}'
    # Check if the path exists, create it if it doesn't
    if not os.path.exists(directory):
        os.makedirs(directory)
    final_image.save(f'{directory}/visualization_batch.png')
    plt.close()



