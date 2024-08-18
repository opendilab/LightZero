

script='source activate cloud-ai-lab && cd /mnt/miaohua/niuyazhe/code/LightZero && pip install -e . && pip3 install ale-py autorom && AutoROM --accept-license && cd /mnt/miaohua/niuyazhe/code/LightZero/zoo/atari/config && python3 -u atari_unizero_config.py'

echo "The final script is: " $script
sco acp jobs create --workspace-name=miaohua \
--aec2-name=miaohua \
--job-name="unizero-pong-200k-optimizehash-save2deepcopy-s0" \
--container-image-url='registry.ms-sc-01.maoshanwangtech.com/ccr_2/aicl-ding-v1:20240719-18h48m08s' \
--training-framework=pytorch \
--enable-mpi \
--worker-nodes=1 \
--worker-spec='N6lS.Iu.I80.1' \
--storage-mount 9063499d-3ffc-11ef-b8ce-929f74fd8884:/mnt/miaohua \
--command="$script"
