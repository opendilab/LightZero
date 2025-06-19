<div id="top"></div>

# LightZero

> LightZero is a lightweight, efficient, and easy-to-understand open-source toolkit combining Monte Carlo Tree Search (MCTS) and deep Reinforcement Learning (RL).  
> This repository includes our **UniZero** implementation and the research on local, adaptive, and Gaussian attention mechanisms for model-based RL. This implementation is part of CSE3000 Research Project.

---

**Updated:** 2025-04-09 LightZero-v0.2.0

## ⚙️ Installation

```bash
git clone https://github.com/opendilab/LightZero.git
cd LightZero
pip install -e .
````

> **Note:** Linux and macOS are officially supported. Windows support is coming soon.

## 🚀 Running UniZero

Train a UniZero agent on Atari Pong:

```bash
    python -m zoo.atari.config.atari_unizero_vanilla_config \
      --env PongNoFrameskip-v4 \
      --seed 1 
```

Train a Local UniZero with window size 7 agent on Atari Pong:

```bash
    python -m zoo.atari.config.atari_unizero_local_config \
      --env PongNoFrameskip-v4 \
      --seed 1 \
      --local_window_size 7
```


Train a Gaussian UniZero on Atari Pong (Params set on the config)
```bash
    python -m zoo.atari.config.atari_unizero_gaussian_config \
      --env PongNoFrameskip-v4 \
      --seed 1 
```

Train an Adaptive UniZero with initial span 6 on Atari Pong 
```bash
    python -m zoo.atari.config.atari_unizero_gaussian_config \
      --env PongNoFrameskip-v4 \
      --seed 1 \
      --init_adaptive_span 6
```


For other UniZero configs, see:

```
zoo/atari/config/atari_unizero_vanilla_config.py
zoo/atari/config/atari_unizero_adaptive_config.py
zoo/atari/config/atari_unizero_gaussian_config.py
zoo/atari/config/atari_unizero_local_config.py
```

## 🔬 Attention Mechanisms Research

We implement and evaluate three attention variants within the UniZero world model:

* **Local Attention** (fixed window)
* **Adaptive Span** (soft, learnable window via ramp mask)
* **Gaussian Adaptive Attention** (soft bell-shaped mask with learnable center μₕ and width σₕ)

**Code organization:**

* **Transformer & Attention**
  `lzero/model/unizero_world_models/modeling/`
  (self-attention implementations, span parametrizations, Gaussian masks)

* **UniZero Policy**
  `lzero/policy/unizero.py`

* **Analysis & Plotting**
  `analysis/`
  (learning-curve plots, attention-map visualizations, ablation scripts)

* **Utilities**
  `lzero/model/unizero_world_models/visualize_utils.py`
  `lzero/model/unizero_world_models/attention_maps.py`
  (frame-sequence and reward/value/policy visualizers)

## 📄 Citation

If you use UniZero or its attention research, please cite:

```bibtex
@article{pu2024unizero,
  title={UniZero: Generalized and Efficient Planning with Scalable Latent World Models},
  author={Pu, Yuan and Niu, Yazhe and Ren, Jiyuan and Yang, Zhenjie and Li, Hongsheng and Liu, Yu},
  journal={arXiv preprint arXiv:2406.10667},
  year={2024}
}
```

## 🏷️ License

This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

<p align="right">(<a href="#top">Back to top</a>)</p>
```
