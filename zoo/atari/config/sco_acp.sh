sco acp jobs create --workspace-name=fb1861da-1c6c-42c7-87ed-e08d8b314a99 \
        --aec2-name=eb37789e-90bb-418d-ad4a-19ce4b81ab0c \
        --job-name=nyz-test \
        --container-image-url='registry.cn-sh-01.sensecore.cn/basemodel-ccr/aicl-nyz-fenxuan-v1:20240820-22h31m28s' \
        --training-framework=pytorch \
        --enable-mpi \
        --worker-nodes=1 \
	--worker-spec='N2lS.Ii.I60.4' \
        --storage-mount 6f8b7bf6-c313-11ed-adcf-92dd2c58bebc:/mnt/afs \
        --command='echo "husbian";sleep inf'
