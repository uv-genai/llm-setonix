# Distributed inference with vLLM

Works with Singularity, Apptainer and Podman the below instructions apply
to Singularity and Apptainer which both work with the same command line, with
Singularity being way faster to execute commands.

Distributed inference is achieved through [Ray](https://www.ray.io/) which
logically aggregates all the computational resources and memory and allows
remote execution of Python code on both CPU and GPU.
The Ray engine is a generic execution engine enabling any kind of distributed
processing beyond serving LLMs.

## 1. Download container

`singularity pull docker://rocm/vllm:rocm6.3.1_instinct_vllm0.8.3_20250415`

Creates the file `vllm_rocm6.3.1_instinct_vllm0.8.3_20250415.sif`

The container includes both Ray and vLLM.

It is better to also map the `/app` folder to a local foldeer to avoid
warnings and to store log files, so add `--bind <local path>:/app` to
the command line, I am using `~/tmp/apptainer/app`.

IMPORTANT: `unset ROCR_VISIBLE_DEVICES` to avoid issues, in general
`HIP_VISIBLE_DEVICES` should be used but it's better not to set any
environment variable beacuse the discovery will happen automatically.

## 2. Add support for Ray dashboard

`singularity exec -H $PWD ./vllm_rocm6.3.1_instinct_vllm0.8.3_20250415.sif pip install ray[default]`

This allows access to the Ray dashboard through `http://nid002220:8265` where
`nid002220` is the head node.

If you want to get more advanced reporting you might want to install `py-spy` and
`memray`, but in any case it is better to integrate with Grafana and Prometheus for
an optimal experience.

## 3. Run Ray on two nodes.

Assuming you have allocated two nodes `nid002220` and `nid002222`
through `salloc`:

1. run ray on the head node (`nid002220`):
   `singularity exec -H $PWD ./vllm_rocm6.3.1_instinct_vllm0.8.3_20250415.sif ray start --num-gpus 8 --head --dashboard-host=0.0.0.0`
2. run ray on all the other worker nodes, only one in this case, specifying the IP address of the head node:
   `singularity exec -H $PWD ./vllm_rocm6.3.1_instinct_vllm0.8.3_20250415.sif ray start --num-gpus 8 --address 10.253.133.93:6379`

Record the IP addresses returned by ray and *on each node* execute:
`export LLVM_HOST_IP=<local ray ip address>`

## 4. (optional) Download the model to serve

``` 
huggingface-cli login
hugginface-cli download Qwen/Qwen3-30B-A3B
```

Make sure you have `hf_ext` installed before downloading: `pip3 install hf_xet`,
Add `--break-system-packages` if not using a virtual env.

The models are installed by default under `.cache/huggingface/hub`; due to the size
of the models it might be a good idea to download them to Lustre and create a symlink.
It is also possible to download a model to a local directory by specifying `--local-dir` 
on the command line.

If you do not download the model, it will be downloaded when `vllm` is executed.

## 4. Run llvm on any node.

```
singularity exec -H $PWD --bind $HOME/tmp/apptainer/app:/app \
  --bind $HOME/.cache/huggingface:/root/.cache/huggingface \
  ./vllm_rocm6.3.1_instinct_vllm0.8.3_20250415.sif \
  vllm serve Qwen/Qwen3-30B-A3B --tensor-parallel-size 8 --pipeline-parallel-size 2 \
  --distributed-executor-backend ray
```
The `--bind $HOME/.cache/huggingface:/root/.cache/huggingface` is not strictly required but
allows to map the `huggingface` folder to any directory.

## 5. Access the OpenAI-compatible service

The OpenAI API can be accessed through the URL `http://nid002220:8000/v1/completions`.

E.g.

```
curl http://nid002220:8000/v1/completions -H "Content-Type: application/json" \
-d '{"input": "What is the capital of France?", "temperature": 0}' | json_pp
```

## (optional) Access the Ray dashboard

Connect to `http://nid002220:8265` to display the Ray dashboard.
Use a tunnel to connect from remote:
`ssh -L 8625:nid002220:8625 <username>@setonix.pawsey.org.au`

## Other toolkits

TODO

[Benchmarks](https://www.inferless.com/learn/exploring-llms-speed-benchmarks-independent-analysis)
