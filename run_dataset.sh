CUDA_VISIBLE_DEVICES=2 \
    python ./main_dataset.py \
    dataset.name=humaneval \
    dataset.size=tiny \
    method=cd_chef \
    method.model=llama-3.1-8b-instruct \
    method.amateur_model=llama-3.2-1b \
    method.generate.temperature=0 \
    method.gamma=5 \
    method.llm.max_model_len=2048
