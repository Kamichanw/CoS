python ./main_dataset.py \
    dataset.name=humaneval \
    dataset.size=5 \
    method=we \
    method.model=Llama-2-7b-hf \
    method.extra_model=vicuna-7b-v1.5 \
    method.gamma=5 \
    method.lambda=0.5