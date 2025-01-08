CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_WORKER_MULTIPROC_METHOD=spawn

task=Attributing
model_name_or_path=meta-llama/Meta-Llama-3.1-70B-Instruct
# model_name_or_path=meta-llama/Meta-Llama-3.1-8B-Instruct
# model_name_or_path=gemini-1.5-pro
# model_name_or_path=THUDM/glm-4-9b-chat
# model_name_or_path=microsoft/Phi-3.5-mini-instruct
# model_name_or_path=Qwen/Qwen2.5-7B-Instruct

# domain=Medicine
# use_yarn=True
# under_32k_only=True
# over_32k_only=True

use_yarn=${use_yarn:-False}
under_32k_only=${under_32k_only:-False}
over_32k_only=${over_32k_only:-False}

if [ "$under_32k_only" = "True" ] && [ "$over_32k_only" = "True" ]; then
    echo "Error: Both under_32k_only and over_32k_only cannot be True simultaneously."
    exit 1
fi

cmd="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python inference.py \
    --task $task \
    --model_name_or_path $model_name_or_path"

if [ "$use_yarn" = "True" ]; then
    cmd="$cmd --use_yarn"
fi
if [ "$under_32k_only" = "True" ]; then
    cmd="$cmd --under_32k_only"
fi
if [ "$over_32k_only" = "True" ]; then
    cmd="$cmd --over_32k_only"
fi
if [ -n "$domain" ]; then
    cmd="$cmd --domain $domain"
fi
if [ -n "$cache_dir" ]; then
    cmd="$cmd --cache_dir $cache_dir"
fi

cmd="$cmd --command \"$cmd\""
eval $cmd