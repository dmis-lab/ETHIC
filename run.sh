CUDA_VISIBLE_DEVICES=1

# arguments
task=Attributing
model_name_or_path=meta-llama/Meta-Llama-3.1-8B-Instruct
cache_dir=""

cmd="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python inference.py \
    --task $task \
    --model_name_or_path $model_name_or_path"

if [ -n "$cache_dir" ]; then
    cmd="$cmd --cache_dir $cache_dir"
fi

eval $cmd