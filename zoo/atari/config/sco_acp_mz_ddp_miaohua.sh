

script='source activate cloud-ai-lab && cd /mnt/miaohua/niuyazhe/code/LightZero && pip install -e . && pip3 install ale-py autorom && AutoROM --accept-license && cd /mnt/miaohua/niuyazhe/code/LightZero/ && python -m torch.distributed.launch --nproc_per_node=2 ./zoo/atari/config/atari_muzero_multigpu_ddp_config.py'

echo "The final script is: " $script
sco acp jobs create --workspace-name=miaohua \
    --aec2-name=miaohua \
    --job-name="muzero-pong-ddp-2gpu_s0" \
    --container-image-url='registry.ms-sc-01.maoshanwangtech.com/ccr_2/aicl-ding-v1:20240719-18h48m08s' \
    --training-framework=pytorch \
    --enable-mpi \
    --worker-nodes=2 \
    --worker-spec='N6lS.Iu.I80.1' \
    --storage-mount 9063499d-3ffc-11ef-b8ce-929f74fd8884:/mnt/miaohua \
    --command="$script"

# --job-name="unizero-pong-200k-origin-s0" \
# --job-name="unizero-pong-200k-optimizehash-4deepcopy-s0" \
# --job-name="unizero-pong-200k-optimizehash-1deepcopy-init-infer-s0" \

