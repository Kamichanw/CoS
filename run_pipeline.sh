read -p "Enter the run name (default to be 'test'): " run_name
run_name="${run_name:=test}"
run_dir=results/$run_name/$(date "+%Y-%m-%d/%H-%M-%S")
mkdir -p $run_dir

python -u ./pipeline.py \
    run_name=test \
	visible_devices=[0] \
	\
    methods=[we_sd,we_chef] \
    +dataset.name=[humaneval] \
    +dataset.size=5 \
    \
    +we.model=Llama-2-7b-hf \
    +we.extra_model=vicuna-7b-v1.5 \
    +we.lambda=0.5 \
    +we.gamma=5 \
    \
    +we_sd.model=Llama-2-7b-hf \
    +we_sd.extra_model=vicuna-7b-v1.5 \
    +we_sd.lambda=0.5 \
    +we_sd.gamma=5 \
    \
    +we_chef.model=Llama-2-7b-hf \
    +we_chef.extra_model=vicuna-7b-v1.5 \
    +we_chef.lambda=0.5 \
    +we_chef.gamma=5 \
    \
    hydra.run.dir=$run_dir