

script='source activate base &&  export HTTPS_PROXY=http://172.16.1.135:3128/ && cd /mnt/afs/niuyazhe/code/LightZero && pip install -e . -i  https://pkg.sensetime.com/repository/pypi-proxy/simple/ && pip3 install ale-py autorom && AutoROM --accept-license && python -m torch.distributed.launch --nproc_per_node=2 ./zoo/atari/config/atari_muzero_multigpu_ddp_config.py'


echo "The final script is: " $script

sco acp jobs create --workspace-name=df42ac16-77cf-4cfe-a3ce-e89e317bdf20 \
    --aec2-name=ea2d41fe-274a-43b2-b562-70c0b7d396a2 \
    --job-name="muzero-pong-ddp-2gpu_s0" \
    --container-image-url='registry.cn-sh-01.sensecore.cn/basemodel-ccr/aicl-b27637a9-660e-4927:20231222-17h24m12s' \
    --training-framework=pytorch \
    --enable-mpi \
    --worker-nodes=1 \
    --worker-spec='N2lS.Ii.I60.2' \
    --storage-mount 6f8b7bf6-c313-11ed-adcf-92dd2c58bebc:/mnt/afs \
    --command="$script"
