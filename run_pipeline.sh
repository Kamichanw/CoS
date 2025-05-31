read -p "Enter the run name (default to be 'test'): " run_name
run_name="${run_name:=test}"
run_dir=results/$run_name/$(date "+%Y-%m-%d/%H-%M-%S")
mkdir -p $run_dir

python -u ./pipeline.py \
    run_name=test \
	visible_devices=[0] \
	\
    methods=[we,we_sd,we_chef,cd,cd_sd,cd_chef] \
    +dataset.name=[humaneval,gsm8k,mmlu,cnndm] \
    +dataset.size=50 \
    \
    +we.model=Llama-2-7b-hf \
    +we.extra_model=vicuna-7b-v1.5 \
    +we.lambda=[0.1,0.3,0.7,0.9] \
    +we.gamma=5 \
    \
    +we_sd.model=Llama-2-7b-hf \
    +we_sd.extra_model=vicuna-7b-v1.5 \
    +we_sd.lambda=[0.1,0.3,0.7,0.9] \
    +we_sd.gamma=5 \
    \
    +we_chef.model=Llama-2-7b-hf \
    +we_chef.extra_model=vicuna-7b-v1.5 \
    +we_chef.lambda=[0.1,0.3,0.7,0.9] \
    +we_chef.gamma=5 \
    \
    +cd.model=Llama-2-7b-hf \
    +cd.extra_model=llama-68m \
    +cd.alpha=[0.2,0.3,0.4,0.5] \
    +cd.gamma=5 \
    \
    +cd_sd.model=Llama-2-7b-hf \
    +cd_sd.extra_model=llama-68m  \
    +cd_sd.alpha=[0.2,0.3,0.4,0.5] \
    +cd_sd.gamma=5 \
    \
    +cd_chef.model=Llama-2-7b-hf \
    +cd_chef.extra_model=llama-68m  \
    +cd_chef.alpha=[0.2,0.3,0.4,0.5] \
    +cd_chef.gamma=[1,5] \
    \
    hydra.run.dir=$run_dir