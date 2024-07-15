# Welcome to LightZero Docs

[![Documentation Status](https://readthedocs.org/projects/lightzero-docs/badge/?version=latest)](https://lightzero-docs.readthedocs.io/en/latest/?badge=latest)

[LightZero](https://arxiv.org/pdf/2310.08348.pdf) is a lightweight, efficient, and easy-to-understand open-source algorithm toolkit that combines Monte Carlo Tree Search (MCTS) and Deep Reinforcement Learning (RL).

For those interested in customizing environments and algorithms, we provide relevant guides:

- [Customize Environments](https://github.com/opendilab/LightZero/blob/main/docs/source/tutorials/envs/customize_envs.md)
- [Customize Algorithms](https://github.com/opendilab/LightZero/blob/main/docs/source/tutorials/algos/customize_algos.md)
- [How to Set Configuration Files?](https://github.com/opendilab/LightZero/blob/main/docs/source/tutorials/config/config.md)
- [Logging and Monitoring System](https://github.com/opendilab/LightZero/blob/main/docs/source/tutorials/logs/logs.md)

Should you have any questions, feel free to contact us for support.


# Local Docs Generation
```bash
# step 1: install
cd LightZero
pip install -r requirements-doc.txt
# step 2: compile docs
cd LightZero/docs/en/source
make live
# step 3: open http://127.0.0.1:8000 in your browser, and explore it!
```

# Citing LightZero-docs
```latex
@misc{ding,
    title={{LightZero-docs:} A Unified Benchmark for Monte Carlo Tree Search in General Sequential Decision Scenarios},
    author={LightZero-docs Contributors},
    publisher = {GitHub},
    howpublished = {\url{https://github.com/opendilab/LightZero-docs}},
    year={2023},
}
```
# Contact Us
If you have any questions about documentation, please add a new issue or contact `opendilab@pjlab.org.cn`

# License

LightZero-Docs released under the Apache 2.0 license
