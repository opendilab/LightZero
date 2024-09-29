envs=(
    'AlienNoFrameskip-v4'
    'AmidarNoFrameskip-v4'
    'AssaultNoFrameskip-v4'
    'AsterixNoFrameskip-v4'
    'BankHeistNoFrameskip-v4'
    'BattleZoneNoFrameskip-v4'
    'ChopperCommandNoFrameskip-v4'
    'CrazyClimberNoFrameskip-v4'
    'DemonAttackNoFrameskip-v4'
    'FreewayNoFrameskip-v4'
    'FrostbiteNoFrameskip-v4'
    'GopherNoFrameskip-v4'
    'HeroNoFrameskip-v4'
    'JamesbondNoFrameskip-v4'
    'KangarooNoFrameskip-v4'
    'KrullNoFrameskip-v4'
    'KungFuMasterNoFrameskip-v4'
    'PrivateEyeNoFrameskip-v4'
    'RoadRunnerNoFrameskip-v4'
    'UpNDownNoFrameskip-v4'
    'PongNoFrameskip-v4'
    'MsPacmanNoFrameskip-v4'
    'QbertNoFrameskip-v4'
    'SeaquestNoFrameskip-v4'
    'BoxingNoFrameskip-v4'
    'BreakoutNoFrameskip-v4'
)
# envs=(
#     'CrazyClimberNoFrameskip-v4'
#     'PongNoFrameskip-v4'
#     'MsPacmanNoFrameskip-v4'
# )

# one env
# env='AsterixNoFrameskip-v4'
# seed=0
# script='source activate base &&  export HTTPS_PROXY=http://172.16.1.135:3128/ && cd /mnt/afs/niuyazhe/code/LightZero && pip install -e . -i  https://pkg.sensetime.com/repository/pypi-proxy/simple/ && pip3 install ale-py autorom && AutoROM --accept-license && python3 -u /mnt/afs/niuyazhe/code/LightZero/zoo/atari/config/atari_unizero_sgement_config_batch.py --env %q --seed %d'
# script=${script/\%q/$env}
# script=${script/\%d/$seed}
# echo "The final script is: " $script

# batch env: uz表现和ez相当的15env
# envs=(
#     'AlienNoFrameskip-v4'
#     'AmidarNoFrameskip-v4'
#     'AssaultNoFrameskip-v4'
#     'BankHeistNoFrameskip-v4'
#     'BattleZoneNoFrameskip-v4'
#     'ChopperCommandNoFrameskip-v4'
#     'FreewayNoFrameskip-v4'
#     'FrostbiteNoFrameskip-v4'
#     'JamesbondNoFrameskip-v4'
#     'KangarooNoFrameskip-v4'
#     'KrullNoFrameskip-v4'
#     'PrivateEyeNoFrameskip-v4'
#     'MsPacmanNoFrameskip-v4'
#     'SeaquestNoFrameskip-v4'
#     'BoxingNoFrameskip-v4'
# )

# batch env: uz表现不如ez的10env+pong
# envs=(
#     'PongNoFrameskip-v4'
#     'QbertNoFrameskip-v4'
#     'AsterixNoFrameskip-v4'
#     # 'CrazyClimberNoFrameskip-v4'
#     # 'DemonAttackNoFrameskip-v4'
#     # 'GopherNoFrameskip-v4'
#     # 'HeroNoFrameskip-v4'
#     # 'KungFuMasterNoFrameskip-v4'
#     # 'RoadRunnerNoFrameskip-v4'
#     # 'UpNDownNoFrameskip-v4'
#     # 'BreakoutNoFrameskip-v4'
# )
seed=1
for env in "${envs[@]}"; do
    script='source activate base &&  export HTTPS_PROXY=http://172.16.1.135:3128/ && pip cache purge && cd /mnt/afs/niuyazhe/code/LightZero && pip install -e . -i  https://pkg.sensetime.com/repository/pypi-proxy/simple/ && pip3 install ale-py autorom && AutoROM --accept-license && python3 -u /mnt/afs/niuyazhe/code/LightZero/zoo/atari/config/atari_muzero_reanalyze_config_batch.py --env %q --seed %d'
	script=${script/\%q/$env}
    script=${script/\%d/$seed}
	echo "The final script is: " $script


sco acp jobs create --workspace-name=df42ac16-77cf-4cfe-a3ce-e89e317bdf20 \
    --aec2-name=ea2d41fe-274a-43b2-b562-70c0b7d396a2\
    --job-name="mz-H5-seg8-gsl20-brf1-10000-rbs160-rr025-temp025-simnorm-origbuffer-$env-s$seed" \
    --container-image-url='registry.cn-sh-01.sensecore.cn/basemodel-ccr/aicl-b27637a9-660e-4927:20231222-17h24m12s' \
    --training-framework=pytorch \
    --enable-mpi \
    --worker-nodes=1 \
    --worker-spec='N2lS.Ii.I60.1' \
    --storage-mount 6f8b7bf6-c313-11ed-adcf-92dd2c58bebc:/mnt/afs \
    --command="$script"
done


