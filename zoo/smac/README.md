## PYSC2 Env
SMAC (StarCraft Multi-Agent Challenge) is an environment used for multi-agent reinforcement learning research based on the popular real-time strategy game StarCraft II. It provides a suite of tasks where agents need to cooperate to achieve specific objectives. This environment is widely used to benchmark the performance of multi-agent learning algorithms. LightZero use modified pysc2 env (for more maps and agent vs agent training).

### Installation

To install the modified PySC2 environment for more maps and agent vs. agent training, follow these steps:

1. **Install StarCraft II:**
    ```bash
    # Create a conda environment
    conda create -n ace python=3.8
    
    # Activate the environment
    conda activate ace
    
    # Download StarCraft II
    wget https://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip
    
    # Unzip the downloaded file
    unzip SC2.4.10.zip
    
    # Set the SC2 path
    export SC2PATH="StarCraftII"
    ```

2. **Install dependencies:**
    ```bash
    # Install PySC2 and Protobuf
    pip install pysc2 protobuf==3.19.5
    ```

### Testing the Installation

To ensure the environment is correctly installed and functional, you can run the tests provided in the LightZero repository:

1. **Navigate to the LightZero directory:**
    ```bash
    cd LightZero/zoo/smac/envs
    ```

2. **Run the tests using pytest:**
    ```bash
    pytest test_smac_env.py

### Environment Requirements

- **Operating System:** Linux (recommended)
- **Python Version:** >=3.8
- **Dependencies:** 
    - StarCraft II (version 4.10)
    - PySC2
    - Protobuf (version 3.19.5)

Make sure your system meets these requirements before proceeding with the installation.

### DI-engine Baselines

For more information on DI-engine baselines for SMAC, refer to the following link: [DI-engine SMAC Baselines](https://github.com/opendilab/DI-engine/blob/main/dizoo/smac/README.md)
