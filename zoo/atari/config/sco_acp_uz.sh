

# container image full name
image="registry.cn-sh-01.sensecore.cn/lepton-trainingjob/centos7.9-py3.9-h800-cuda11.8-ofed5.8-deepspeed-transformer4.28-mpi4.1.3:v1.0.0-20231130"
# replace '%q' with correspodning env variable in launch script of sensecore container

script='source activate base && cd /mnt/afs/niuyazhe/code/LightZero && pip install -e . && pip3 install ale-py autorom && AutoROM --accept-license && cd /mnt/afs/niuyazhe/code/LightZero/zoo/atari/config && python3 -u atari_unizero_config.py'

echo "The final script is: " $script
# sco acp jobs create --workspace-name=nlp \
# --aec2-name=nlp \
# --job-name="unizero-pong-nlayer2-200k-opthash_only-recur-save-dc_opt-comploss_opt-value-lst_policy-nonrer_kv-update_kv-nocpu-v2_s0" \
# --container-image-url='registry.ms-sc-01.maoshanwangtech.com/basemodel_ccr/aicl-b27637a9-660e-4927:20231222-17h24m12s' \
# --training-framework=pytorch \
# --enable-mpi \
# --worker-nodes=1 \
# --worker-spec='N2lS.Ii.I60.8' \
# --storage-mount ea2d41fe-274a-43b2-b562-70c0b7d396a2:/mnt/afs \
# --command="$script"


srun -p ea2d41fe-274a-43b2-b562-70c0b7d396a2 \
     --workspace-id df42ac16-77cf-4cfe-a3ce-e89e317bdf20 \
     -r N2lS.Ii.I60.8 \
     --framework pytorch \
     --mpi \
     --container-image $image \
     --job-name "nccl-tests" \
     --nodes $1 \
     bash -c "$script"