## Installation and Quick Start Guide

### Installation

LightZero is currently available on PyPI and requires Python version 3.7 or higher.

To install LightZero from PyPI, use the following command:

```bash
pip install LightZero
```

Alternatively, you can install the latest development version of LightZero directly from the GitHub repository with these commands:

```bash
git clone https://github.com/opendilab/LightZero.git
cd LightZero
pip3 install -e .
```

Please note that LightZero currently supports installation on `Linux` and `macOS` platforms only. We are actively working to extend support to the `Windows` platform, and we appreciate your patience during this transition.

### Installation with Docker

We also provide a Dockerfile that sets up an environment with all dependencies needed to run the LightZero library. This Docker image is based on Ubuntu 20.04 and installs Python 3.8, along with other necessary tools and libraries.

Here's how to use the Dockerfile to build a Docker image, run a container from this image, and execute LightZero code inside the container:

1. **Download the Dockerfile**: The Dockerfile is located in the root directory of the LightZero repository. Download this [file](https://github.com/opendilab/LightZero/blob/main/Dockerfile) to your local machine.

2. **Prepare the build context**: Create a new directory on your local machine, move the Dockerfile into this directory, and navigate into the directory. This step helps avoid sending unnecessary files to the Docker daemon during the build process.

    ```bash
    mkdir lightzero-docker
    mv Dockerfile lightzero-docker/
    cd lightzero-docker/
    ```

3. **Build the Docker image**: Use the following command to build the Docker image from within the directory containing the Dockerfile.

    ```bash
    docker build -t ubuntu-py38-lz:latest -f ./Dockerfile .
    ```

4. **Run a container from the image**: Use the following command to start a container from the image in interactive mode with a Bash shell.

    ```bash
    docker run -dit --rm ubuntu-py38-lz:latest /bin/bash
    ```

5. **Execute LightZero code inside the container**: Once inside the container, you can run the example Python script with the following command:

    ```bash
    python ./LightZero/zoo/classic_control/cartpole/config/cartpole_muzero_config.py
    ```

### Quick Start

#### Train a MuZero Agent to Play CartPole

To train a MuZero agent to play [CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/), use the following commands:

```bash
cd LightZero
python3 -u zoo/classic_control/cartpole/config/cartpole_muzero_config.py
```

#### Train a MuZero Agent to Play Pong

To train a MuZero agent to play [Pong](https://gymnasium.farama.org/environments/atari/pong/), use the following commands:

```bash
cd LightZero
python3 -u zoo/atari/config/atari_muzero_config.py
```

#### Train a MuZero Agent to Play TicTacToe

To train a MuZero agent to play [TicTacToe](https://en.wikipedia.org/wiki/Tic-tac-toe), use the following commands:

```bash
cd LightZero
python3 -u zoo/board_games/tictactoe/config/tictactoe_muzero_bot_mode_config.py
```

Feel free to reach out with any questions or issues as you explore LightZero!