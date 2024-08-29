# 🎨 Welcome to LightZero Docs

[![Documentation Status](https://readthedocs.org/projects/lightzero-docs/badge/?version=latest)](https://lightzero-docs.readthedocs.io/en/latest/?badge=latest)

[LightZero](https://arxiv.org/pdf/2310.08348.pdf) is a lightweight, efficient, and easy-to-understand open-source algorithm toolkit that combines Monte Carlo Tree Search (MCTS) and Deep Reinforcement Learning (RL).

The LightZero documentation can be found [here](https://opendilab.github.io/LightZero/). It contains tutorials and the API reference.

For those interested in customizing environments and algorithms, we provide relevant guides:

- [Customize Environments](https://github.com/opendilab/LightZero/blob/main/docs/en/source//tutorials/envs/customize_envs.md)
- [Customize Algorithms](https://github.com/opendilab/LightZero/blob/main/docs/en/source//tutorials/algos/customize_algos.md)
- [How to Set Configuration Files?](https://github.com/opendilab/LightZero/blob/main/docs/en/source//tutorials/config/config.md)
- [Logging and Monitoring System](https://github.com/opendilab/LightZero/blob/main/docs/en/source//tutorials/logs/logs.md)

Should you have any questions, feel free to contact us for support.


# ⚙️ Local Docs Generation
```bash
# step 1: install
cd LightZero
pip install -r requirements-doc.txt
# step 2: compile docs
cd LightZero/docs/source
make live
# step 3: open http://127.0.0.1:8000 in your browser, and explore it!
```

# 🌏 Citing LightZero-docs

```latex
@article{niu2024lightzero,
  title={LightZero: A Unified Benchmark for Monte Carlo Tree Search in General Sequential Decision Scenarios},
  author={Niu, Yazhe and Pu, Yuan and Yang, Zhenjie and Li, Xueyan and Zhou, Tong and Ren, Jiyuan and Hu, Shuai and Li, Hongsheng and Liu, Yu},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}

@article{pu2024unizero,
  title={UniZero: Generalized and Efficient Planning with Scalable Latent World Models},
  author={Pu, Yuan and Niu, Yazhe and Ren, Jiyuan and Yang, Zhenjie and Li, Hongsheng and Liu, Yu},
  journal={arXiv preprint arXiv:2406.10667},
  year={2024}
}

@article{xuan2024rezero,
  title={ReZero: Boosting MCTS-based Algorithms by Backward-view and Entire-buffer Reanalyze},
  author={Xuan, Chunyu and Niu, Yazhe and Pu, Yuan and Hu, Shuai and Liu, Yu and Yang, Jing},
  journal={arXiv preprint arXiv:2404.16364},
  year={2024}
}
```
# Contact Us
If you have any questions about documentation, please add a new issue or contact `opendilab@pjlab.org.cn`

# License

LightZero-Docs released under the Apache 2.0 license
