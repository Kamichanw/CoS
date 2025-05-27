CUDA_VISIBLE_DEVICES=2 \
    python ./main_dataset.py \
    dataset.name=humaneval \
    dataset.size=tiny \
    method=we_chef \
    method.model=Llama-2-7b-hf \
    method.extra_model=llama-68m \
    method.gamma=5 \
    method.lambda=0.5
