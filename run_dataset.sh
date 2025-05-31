python ./main_dataset.py \
    dataset.name=humaneval \
    dataset.size=5 \
    method=cd_chef \
    method.model=Llama-2-7b-hf \
    method.extra_model=llama-68m \
    method.gamma=5 \
    method.alpha=0.1 \