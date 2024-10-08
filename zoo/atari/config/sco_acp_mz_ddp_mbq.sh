

script='source activate base &&  export HTTPS_PROXY=http://172.16.1.135:3128/ && cd /mnt/afs/niuyazhe/code/LightZero && pip install -e . -i  https://pkg.sensetime.com/repository/pypi-proxy/simple/ && pip3 install ale-py autorom && AutoROM --accept-license && python -m torch.distributed.launch --nproc_per_node=2 ./zoo/atari/config/atari_muzero_multigpu_ddp_config.py'

echo "The final script is: " $script

sco acp jobs create --workspace-name=fb1861da-1c6c-42c7-87ed-e08d8b314a99 \
    --aec2-name=eb37789e-90bb-418d-ad4a-19ce4b81ab0c\
    --job-name="muzero-pong-ddp-2gpu_s0" \
    --container-image-url='registry.cn-sh-01.sensecore.cn/basemodel-ccr/aicl-b27637a9-660e-4927:20231222-17h24m12s' \
    --training-framework=pytorch \
    --enable-mpi \
    --worker-nodes=1 \
    --worker-spec='N2lS.Ii.I60.2' \
    --storage-mount 6f8b7bf6-c313-11ed-adcf-92dd2c58bebc:/mnt/afs \
    --command="$script"
