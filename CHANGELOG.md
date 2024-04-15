2024.04.12 (v0.0.5)
- env: MemoryEnv (#197)
- env: mountain_car (#181)
- algo: Gumbel AlphaZero in ctree (#212)
- feature: add ZeroPal and discord link (#209)
- feature: add eval_offline option (#188)
- feature: save the updated searched policy and value to the buffer during reanalyze (#190)
- feature: add muzero visualization (#181)
- feature: add efficientzero tictactoe configs (#204)
- feature: add 2 mcts related iclr2024 papers
- polish: add load pretrained model option in test_game_segment (#194)
- polish: polish _forward_learn() and some data process operations (#191)
- polish: add customization documentation section in readme
- fix: fix sync_gradients and log in DDP settings (#200)
- fix: fix channel_last bug
- fix: fix total_episode_count bug in collector
- fix: fix memory_lightzero_env return bug
- fix: fix obs_max_scale bug in memory_env
- style: add unittest for game_buffer_muzero (#186)

2024.02.08 (v0.0.4)
- polish: add agent configurations & polish replay video saving method (#184)
- polish: polish comments in worker files
- polish: polish comments in tree search files (#185)
- polish: rename mcts_mode to battle_mode_in_simulation_env, add sampled alphazero config for tictactoe (#179)
- polish: polish redundant data squeeze operations (#177)
- polish: polish the continuous action process in sez model
- polish: polish bipedalwalker env
- fix: fix completed value inf bug when zero exists in action_mask in gumbel muzero (#178)
- fix: fix render settings when using gymnasium (#173)
- fix: fix lstm_hidden_size in sampled_efficientzero_model.py
- fix: fix action_mask in bipedalwalker_cont_disc_env, fix device bug in sampled efficientzero (#168)

2023.12.07 (v0.0.3)
- env: MiniGrid env (#110)
- env: Bsuite env (#110)
- env: GoBigger env (#39)
- algo: RND+MuZero (#110)
- algo: Sampled AlphaZero (#141)
- algo: Multi-Agent MuZero/EfficientZero (#39)
- feature: add ctree version of mcts in alphazero (#142)
- feature: upgrade the dependency on gym with gymnasium (#150)
- feature: add agent class to support LightZero's HuggingFace Model Zoo (#163)
- feature: add recent MCTS-related papers in readme (#159)
- feature: add muzero config for connect4 (#107)
- feature: added CONTRIBUTING.md (#119)
- feature: added .gitpod.yml and .gitpod.Dockerfile (#123)
- feature: added contributors subsection in README (#132) 
- feature: added CODE_OF_CONDUCT.md (#127)
- polish: refine comments and render_eval configs for various common envs (#154) (#161)
- polish: polish action_type and env_type, fix test.yml, fix unittest (#160)
- polish: update env and algo tutorial doc (#106)
- polish: polish gomoku env (#141)
- polish: add random_policy support for continuous env (#118)
- polish: polish simulation method of ptree_az (#120)
- polish: polish comments of game_segment_to_array
- fix: fix render method for various common envs (#154) (#161)
- fix: fix gumbel muzero collector bug, fix gumbel typo (#144)
- fix: fix assert bug in game_segment.py (#138)
- fix: fix visit_count_distributions name in muzero_evaluator
- fix: fix mcts and alphabeta bot unittest (#120)
- fix: fix typos in ptree_mz.py (#113)
- fix: fix root_sampled_actions_tmp shape bug in sez ptree
- fix: fix policy utils unittest
- fix: fix typo in readme and add a 'back to top' button in readme (#104) (#109) (#111)
- style: add nips2023 paper link

2023.09.21 (v0.0.2)
- env: MuJoCo env (#50)
- env: 2048 env (#64)
- env: Connect4 env (#63)
- algo: Gumbel MuZero (#22)
- algo: Stochastic MuZero (#64)
- feature: add Dockerfile and its usage instructions (#95)
- feature: add doc about how to customize envs and algos (#78)
- feature: add pytorch ddp support (#68)
- feature: add eps greedy and random collect option in train_muzero_entry (#54)
- feature: add atari visualization option (#40)
- feature: add log_buffer_memory_usage utils (#30)
- polish: polish mcts and ptree_az (#57) (#61)
- polish: polish readme (#36) (#47) (#51) (#77) (#95) (#96)
- polish: update paper notes (#89) (#91)
- polish: polish model and configs (#26) (#27) (#50)
- fix: fix priority bug in muzero collector (#74)
- style: update github action (#71) (#72) (#73) (#81) (#83) (#84) (#90)

2023.04.14 (v0.0.1)