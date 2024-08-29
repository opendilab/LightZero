## 安装和快速入门指南

### 安装

LightZero 目前可通过 PyPI 获取，并需要 Python 3.7 或更高版本。

要从 PyPI 安装 LightZero，请使用以下命令：

```bash
pip install LightZero
```

或者，您可以通过以下命令直接从 GitHub 仓库安装最新的开发版本：

```bash
git clone https://github.com/opendilab/LightZero.git
cd LightZero
pip3 install -e .
```

请注意，LightZero 目前仅支持在 `Linux` 和 `macOS` 平台上安装。我们正在积极工作以扩展对 `Windows` 平台的支持，感谢您在此过渡期间的耐心等待。

### 使用 Docker 安装

我们还提供了一个 Dockerfile，用于设置一个包含运行 LightZero 库所需的所有依赖项的环境。此 Docker 镜像基于 Ubuntu 20.04，并安装了 Python 3.8 以及其他必要的工具和库。

以下是使用 Dockerfile 构建 Docker 镜像、从此镜像运行容器并在容器内执行 LightZero 代码的方法：

1. **下载 Dockerfile**：Dockerfile 位于 LightZero 仓库的根目录。将此 [文件](https://github.com/opendilab/LightZero/blob/main/Dockerfile) 下载到您的本地机器。

2. **准备构建上下文**：在本地机器上创建一个新目录，将 Dockerfile 移动到此目录，并导航到该目录。此步骤有助于在构建过程中避免将不必要的文件发送到 Docker 守护程序。

    ```bash
    mkdir lightzero-docker
    mv Dockerfile lightzero-docker/
    cd lightzero-docker/
    ```

3. **构建 Docker 镜像**：在包含 Dockerfile 的目录中使用以下命令构建 Docker 镜像。

    ```bash
    docker build -t ubuntu-py38-lz:latest -f ./Dockerfile .
    ```

4. **从镜像运行容器**：使用以下命令以交互模式启动一个从镜像生成的容器，并带有 Bash shell。

    ```bash
    docker run -dit --rm ubuntu-py38-lz:latest /bin/bash
    ```

5. **在容器内执行 LightZero 代码**：进入容器后，您可以使用以下命令运行示例 Python 脚本：

    ```bash
    python ./LightZero/zoo/classic_control/cartpole/config/cartpole_muzero_config.py
    ```

### 快速入门

#### 训练 MuZero Agent 玩 CartPole

要训练一个 MuZero agent 来玩 [CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/)，请使用以下命令：

```bash
cd LightZero
python3 -u zoo/classic_control/cartpole/config/cartpole_muzero_config.py
```

#### 训练 MuZero Agent 玩 Pong

要训练一个 MuZero agent 来玩 [Pong](https://gymnasium.farama.org/environments/atari/pong/)，请使用以下命令：

```bash
cd LightZero
python3 -u zoo/atari/config/atari_muzero_config.py
```

#### 训练 MuZero Agent 玩 TicTacToe

要训练一个 MuZero agent 来玩 [TicTacToe](https://en.wikipedia.org/wiki/Tic-tac-toe)，请使用以下命令：

```bash
cd LightZero
python3 -u zoo/board_games/tictactoe/config/tictactoe_muzero_bot_mode_config.py
```

在探索 LightZero 的过程中，如有任何问题或疑问，请随时联系我们！