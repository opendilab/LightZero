# envs=(
#     'AlienNoFrameskip-v4'
#     'AmidarNoFrameskip-v4'
#     'AssaultNoFrameskip-v4'
#     'AsterixNoFrameskip-v4'
#     'BankHeistNoFrameskip-v4'
#     'BattleZoneNoFrameskip-v4'
#     'ChopperCommandNoFrameskip-v4'
#     'CrazyClimberNoFrameskip-v4'
#     'DemonAttackNoFrameskip-v4'
#     'FreewayNoFrameskip-v4'
#     'FrostbiteNoFrameskip-v4'
#     'GopherNoFrameskip-v4'
#     'HeroNoFrameskip-v4'
#     'JamesbondNoFrameskip-v4'
#     'KangarooNoFrameskip-v4'
#     'KrullNoFrameskip-v4'
#     'KungFuMasterNoFrameskip-v4'
#     'PrivateEyeNoFrameskip-v4'
#     'RoadRunnerNoFrameskip-v4'
#     'UpNDownNoFrameskip-v4'
#     'PongNoFrameskip-v4'
#     'MsPacmanNoFrameskip-v4'
#     'QbertNoFrameskip-v4'
#     'SeaquestNoFrameskip-v4'
#     'BoxingNoFrameskip-v4'
#     'BreakoutNoFrameskip-v4'
# )

# envs=(
#     'AlienNoFrameskip-v4'
#     'AmidarNoFrameskip-v4'
#     'AssaultNoFrameskip-v4'
#     'AsterixNoFrameskip-v4'
#     'ChopperCommandNoFrameskip-v4'
#     'DemonAttackNoFrameskip-v4'
#     'KangarooNoFrameskip-v4'
#     'KrullNoFrameskip-v4'
#     'KungFuMasterNoFrameskip-v4'
#     'RoadRunnerNoFrameskip-v4'
#     'UpNDownNoFrameskip-v4'
# )

# one env
# env='AsterixNoFrameskip-v4'
# seed=0
# script='source activate base &&  export HTTPS_PROXY=http://172.16.1.135:3128/ && cd /mnt/afs/niuyazhe/code/LightZero && pip install -e . -i  https://pkg.sensetime.com/repository/pypi-proxy/simple/ && pip3 install ale-py autorom && AutoROM --accept-license && python3 -u /mnt/afs/niuyazhe/code/LightZero/zoo/atari/config/atari_unizero_sgement_config_batch.py --env %q --seed %d'
# script=${script/\%q/$env}
# script=${script/\%d/$seed}
# echo "The final script is: " $script

# batch env
envs=(
    'PongNoFrameskip-v4'
    'QbertNoFrameskip-v4'
    'AsterixNoFrameskip-v4'
)
seed=0
for env in "${envs[@]}"; do
    script='source activate base &&  export HTTPS_PROXY=http://172.16.1.135:3128/ && cd /mnt/afs/niuyazhe/code/LightZero && pip install -e . -i  https://pkg.sensetime.com/repository/pypi-proxy/simple/ && pip3 install ale-py autorom && AutoROM --accept-license && python3 -u /mnt/afs/niuyazhe/code/LightZero/zoo/atari/config/atari_unizero_sgement_config_batch.py --env %q --seed %d'
	script=${script/\%q/$env}
    script=${script/\%d/$seed}
	echo "The final script is: " $script

sco acp jobs create --workspace-name=fb1861da-1c6c-42c7-87ed-e08d8b314a99 \
    --aec2-name=eb37789e-90bb-418d-ad4a-19ce4b81ab0c\
    --job-name="uz-nlayer2-H10-seg8-gsl20-brf1-5-$env-s$seed" \
    --container-image-url='registry.cn-sh-01.sensecore.cn/basemodel-ccr/aicl-b27637a9-660e-4927:20231222-17h24m12s' \
    --training-framework=pytorch \
    --enable-mpi \
    --worker-nodes=1 \
    --worker-spec='N2lS.Ii.I60.1' \
    --storage-mount 6f8b7bf6-c313-11ed-adcf-92dd2c58bebc:/mnt/afs \
    --command="$script"
done


# --job-name="uz-nlayer2-H5-$env-s$seed" \
