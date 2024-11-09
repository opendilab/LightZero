import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def compute_normalized_mean_and_median(
    random_scores: list[float],
    human_scores: list[float],
    algo_scores: list[float]
) -> tuple[float, float]:
    """
    Computes the normalized mean and median based on random, human, and algorithm scores.

    Args:
        random_scores (list[float]): List of random scores for each game.
        human_scores (list[float]): List of human scores for each game.
        algo_scores (list[float]): List of algorithm scores for each game.

    Returns:
        tuple[float, float]:
            - The mean of the normalized scores.
            - The median of the normalized scores.

    Raises:
        ValueError: If any list is empty or if the lengths of the input lists do not match.
    """
    if not random_scores or not human_scores or not algo_scores:
        raise ValueError("Input score lists must not be empty.")
    if len(random_scores) != len(human_scores) or len(human_scores) != len(algo_scores):
        raise ValueError("Input score lists must have the same length.")

    # Calculate normalized scores
    normalized_scores = [
        (algo_score - random_score) / (human_score - random_score)
        if human_score != random_score else 0
        for random_score, human_score, algo_score in zip(random_scores, human_scores, algo_scores)
    ]

    # Compute mean and median of the normalized scores
    normalized_mean = np.mean(normalized_scores)
    normalized_median = np.median(normalized_scores)

    return normalized_mean, normalized_median


def plot_normalized_scores(
    algorithms: list[str],
    means: list[float],
    medians: list[float],
    filename: str = "normalized_scores.png"
) -> None:
    """
    Plots a bar chart for normalized mean and median values for different algorithms.

    Args:
        algorithms (list[str]): List of algorithm names.
        means (list[float]): List of normalized mean values.
        medians (list[float]): List of normalized median values.
        filename (str, optional): Filename to save the plot (default is 'normalized_scores.png').

    Returns:
        None

    Raises:
        ValueError: If lists of algorithms, means, or medians have different lengths.

    Example usage:
        algorithms = ["Algorithm A", "Algorithm B", "Algorithm C"]
        means = [0.75, 0.85, 0.60]
        medians = [0.70, 0.80, 0.65]
        plot_normalized_scores(algorithms, means, medians)
    """
    if not (len(algorithms) == len(means) == len(medians)):
        raise ValueError("Algorithms, means, and medians lists must have the same length.")

    # Set a style suited for academic papers (muted, professional colors)
    sns.set(style="whitegrid")

    x = np.arange(len(algorithms))  # The label locations
    width = 0.35  # Width of the bars

    # Set up the figure with a larger size (good for academic papers)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define color palette for consistency and readability
    mean_color = sns.color_palette("muted")[0]  # Muted blue
    median_color = sns.color_palette("muted")[1]  # Muted orange

    # Plotting bars for mean and median
    bars_mean = ax.bar(x - width / 2, means, width, label='Normalized Mean', color=mean_color)
    bars_median = ax.bar(x + width / 2, medians, width, label='Normalized Median', color=median_color)

    # Add labels, title, and custom x-axis tick labels
    ax.set_ylabel('Scores', fontsize=14)
    ax.set_title('Human Normalized Score (Atari 100k)', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, fontsize=12)
    ax.legend(fontsize=12)

    # Add grid lines for better readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)

    # Attach a text label above each bar displaying its height
    def attach_labels(bars):
        for bar in bars:
            height = bar.get_height()
            # Annotate with precision of two decimal places
            ax.annotate(f'{height:.2f}',  # Text to display
                        xy=(bar.get_x() + bar.get_width() / 2, height),  # Position
                        xytext=(0, 3),  # Offset from the top of the bar
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=11)

    attach_labels(bars_mean)
    attach_labels(bars_median)

    # Adjust layout for tight fit (avoids cutting off labels)
    fig.tight_layout()

    # Save the plot as a high-resolution image suitable for publications
    plt.savefig(filename, dpi=300)
    print(f"Plot saved as {filename}")
    plt.close()


# Scores for Atari 100k 26 games
random_scores = [
    227.8,   # Alien
    5.8,     # Amidar
    222.4,   # Assault
    210.0,   # Asterix
    14.2,    # BankHeist
    2360.0,  # BattleZone
    0.1,     # Boxing
    1.7,     # Breakout
    811.0,   # ChopperCommand
    10780.5, # CrazyClimber
    152.1,   # DemonAttack
    0.0,     # Freeway
    65.2,    # Frostbite
    257.6,   # Gopher
    1027.0,  # Hero
    29.0,    # Jamesbond
    52.0,    # Kangaroo
    1598.0,  # Krull
    258.5,   # KungFuMaster
    307.3,   # MsPacman
    -20.7,   # Pong
    24.9,    # PrivateEye
    163.9,   # Qbert
    11.5,    # RoadRunner
    68.4,    # Seaquest
    533.4    # UpNDown
]

human_scores = [
    7127.7,  # Alien
    1719.5,  # Amidar
    742.0,   # Assault
    8503.3,  # Asterix
    753.1,   # BankHeist
    37187.5, # BattleZone
    12.1,    # Boxing
    30.5,    # Breakout
    7387.8,  # ChopperCommand
    35829.4, # CrazyClimber
    1971.0,  # DemonAttack
    29.6,    # Freeway
    4334.7,  # Frostbite
    2412.5,  # Gopher
    30826.4, # Hero
    302.8,   # Jamesbond
    3035.0,  # Kangaroo
    2665.5,  # Krull
    22736.3, # KungFuMaster
    6951.6,  # MsPacman
    14.6,    # Pong
    69571.3, # PrivateEye
    13455.0, # Qbert
    7845.0,  # RoadRunner
    42054.7, # Seaquest
    11693.2  # UpNDown
]

ez_scores = [
    808.5,   # Alien
    149,     # Amidar
    1263,    # Assault
    25558,   # Asterix
    351,     # BankHeist
    13871,   # BattleZone
    53,      # Boxing
    414,     # Breakout
    1117,    # ChopperCommand
    83940,   # CrazyClimber
    13004,   # DemonAttack
    22,      # Freeway
    296,     # Frostbite
    3260,    # Gopher
    9315,    # Hero
    517,     # Jamesbond
    724,     # Kangaroo
    5663,    # Krull
    30945,   # KungFuMaster
    1281,    # MsPacman
    20,      # Pong
    97,      # PrivateEye
    13782,   # Qbert
    17751,   # RoadRunner
    1100,    # Seaquest
    17264    # UpNDown
]

mz_scores = [
    530.0,  # Alien
    39,     # Amidar
    500,    # Assault
    1734,   # Asterix
    193,    # BankHeist
    7688,   # BattleZone
    15,     # Boxing
    48,     # Breakout
    1350,   # ChopperCommand
    56937,  # CrazyClimber
    3527,   # DemonAttack
    22,     # Freeway
    255,    # Frostbite
    1256,   # Gopher
    3095,   # Hero
    88,     # Jamesbond
    63,     # Kangaroo
    4891,   # Krull
    18813,  # KungFuMaster
    1266,   # MsPacman
    -7,     # Pong
    56,     # PrivateEye
    3952,   # Qbert
    2500,   # RoadRunner
    208,    # Seaquest
    2897    # UpNDown
]

mz_ssl_scores = [
    700,     # Alien
    90,      # Amidar
    600,     # Assault
    1400,    # Asterix
    33,      # BankHeist
    7587,    # BattleZone
    20,      # Boxing
    4,       # Breakout
    2050,    # ChopperCommand
    26060,   # CrazyClimber
    4601,    # DemonAttack
    12,      # Freeway
    260,     # Frostbite
    646,     # Gopher
    9315,    # Hero
    300,     # Jamesbond
    600,     # Kangaroo
    2700,    # Krull
    25100,   # KungFuMaster
    1410,    # MsPacman
    -15,     # Pong
    100,     # PrivateEye
    4700,    # Qbert
    3400,    # RoadRunner
    566,     # Seaquest
    5213     # UpNDown
]

unizero_scores = [
    1000,    # Alien
    96,      # Amidar
    609,     # Assault
    1016,    # Asterix
    50,      # BankHeist
    11410,   # BattleZone
    7,       # Boxing
    12,      # Breakout
    3205,    # ChopperCommand
    13666,   # CrazyClimber
    1001,    # DemonAttack
    7,       # Freeway
    310,     # Frostbite
    1153,    # Gopher
    8005,    # Hero
    305,     # Jamesbond
    1285,    # Kangaroo
    3484,    # Krull
    15600,   # KungFuMaster
    1927,    # MsPacman
    18,      # Pong
    1048,    # PrivateEye
    3056,    # Qbert
    11000,   # RoadRunner
    620,     # Seaquest
    4523     # UpNDown
]

# Calculate normalized mean and median for each algorithm
ez_mean, ez_median = compute_normalized_mean_and_median(random_scores, human_scores, ez_scores)
mz_mean, mz_median = compute_normalized_mean_and_median(random_scores, human_scores, mz_scores)
mz_ssl_mean, mz_ssl_median = compute_normalized_mean_and_median(random_scores, human_scores, mz_ssl_scores)
unizero_mean, unizero_median = compute_normalized_mean_and_median(random_scores, human_scores, unizero_scores)

# Print the results
print(f"EZ - Normalized Mean: {ez_mean}, Normalized Median: {ez_median}")
print(f"MZ - Normalized Mean: {mz_mean}, Normalized Median: {mz_median}")
print(f"MZ with SSL - Normalized Mean: {mz_ssl_mean}, Normalized Median: {mz_ssl_median}")
print(f"UniZero - Normalized Mean: {unizero_mean}, Normalized Median: {unizero_median}")

# Plot the normalized means and medians for each algorithm
algorithms = ['MZ', 'MZ with SSL', 'UniZero']
means = [mz_mean, mz_ssl_mean, unizero_mean]
medians = [mz_median, mz_ssl_median, unizero_median]

# Save the plot as a PNG file
plot_normalized_scores(algorithms, means, medians, filename="atari100k_normalized_scores_3algo.png")