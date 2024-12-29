# ETHIC: Evaluating Large Language Models on Long-Context Tasks with High Information Coverage

<p align="center">
    📃 <a href="https://arxiv.org/abs/2410.16848" target="_blank">Paper</a> | 🤗 <a href="https://huggingface.co/datasets/dmis-lab/ETHIC" target="_blank">Dataset</a>
</p>

## 📋 Introduction
**ETHIC** is a long-context benchmark designed to assess whether LLMs can fully utilize the provided information. ETHIC comprises tasks with high **Information Coverage (IC)** scores (~91%), i.e. the proportion of input context necessary for answering queries.   

![](figs/long_context_figure.pdf)

## ⏩ Quickstart
To use our dataset directly, simply download it using 🤗 Datasets:

```python
from datasets import load_dataset

task = "Recalling" # Choose from "Recalling", "Summarizing", "Organizing", "Attributing"
dataset = load_dataset("dmis-lab/ETHIC", task)["test"]
```

To evaluate models using our benchmark, please follow the steps below.

### Setup
We recommend using the following versions to ensure compatibility.
* PyTorch 2.4.0
* Cuda 12.1
```shell
# create a new environment
conda create -n ethic python==3.9.19
conda activate ethic

# install required packages
pip install -r requirements.txt
```

### Inference
Make sure to prepare your OpenAI API key (or other keys for authorization) in _api_config.py_, since we utilize `gpt-4o` for evaluation in the _Summarizing_ task.
```shell
CUDA_VISIBLE_DEVICES=1

# arguments
task=Attributing # Recalling, Summarizing, Organizing, Attributing
model_name_or_path=meta-llama/Meta-Llama-3.1-8B-Instruct
cache_dir=""

cmd="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python inference.py \
    --task $task \
    --model_name_or_path $model_name_or_path"

if [ -n "$cache_dir" ]; then
    cmd="$cmd --cache_dir $cache_dir"
fi

eval $cmd
```

## Citation
```
@article{lee2024ethic,
  title={ETHIC: Evaluating Large Language Models on Long-Context Tasks with High Information Coverage},
  author={Lee, Taewhoo and Yoon, Chanwoong and Jang, Kyochul and Lee, Donghyeon and Song, Minju and Kim, Hyunjae and Kang, Jaewoo},
  journal={arXiv preprint arXiv:2410.16848},
  year={2024}
}
```
