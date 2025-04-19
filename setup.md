# Setup a GenAI environment

This document shows how to setup a GenAI development environment
on the GPU nodes of the Setonix Cray-EX supercomputer at the 
Pawsey Surpercomputing Centre.
Since all the packages need to be built and installed from scratch
in the home directory the instructions should be generic enough
to apply to any other Linux cluster.

The main component of an GenAI environment is the inference platform
to server the models.

In this document we are going to cover the installation and usage of
the following serving platforms:
1. ollama
2. llama.cpp
3. SGLang
4. vLLM

Given the fac that the currently biggest model (DeepSeek-r1) can
ran on a single node at a reasonable speed it might not be
required to use SGLang or vLLM to distribute a model on
multiple nodes.

All the platforms expose both a chat and an OpenAI compatible
endpoint.

SGLang does not work at the moment with the latest version of 
PyTorch and Python 3.13.

## 1. Install Python

A recent version of Python is required to guarantee that all
the packages work properly.

At the time of this writing Python 3.13.3 is the latest version.

Since on shared systems Python cannot be installed using a package manager
it needs to be build from source code.

Procedure to follow on Cray systems after having downloaded and unpacked
the python source archive downloaded from https://www.python.org/downloads/source/
to install Python and `pip` nder `~/.local`:

1. `module load ProgEnv-gnu`
2. `cd Python-3.13.3`
3. `./configure --prefix=~/.local -enable-optimizations --with-lto=full --prefix=$PWD/run --with-ensurepip --libdir=$PWD/run/lib`
4. `make -j $(nproc)`
5. `make install`

In order to find the `_posixsubprocess` package you need to make the `lib-dynload` directory
accessible from `~/.local/lib/python3.13`:

`ln -sf ~/.local/lib64/python3.13/lib-dynload ~/.local/lib/python3.13/lib-dynload`.


## 2. Install Ollama

Download the version for your system from here: https://github.com/ollama/ollama/blob/main/docs/linux.md

### Basic installation

Install first the default version:

`curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz`

`tar xzf ollama-linux-amd64.tgz`

Assuming you are installing Ollama under `~/.local`:

`cp bin/ollama ~/.local/bin`
`cp -r lib/ollama ~/.local/lib`

Try it out:

`ollama serve &`
`ollama run --verbose deepseek-r1:32b`

On a 128 core (2x AMD Milan 64 core CPUs) with 256 GB of memory it is possible to run models
up to 32 billion parameters at above 4 tokens/second without any optimisation.


### AMD GPU version

After having installed the cpu/NVIDIA version, download the ROCm version:

`curl -L https://ollama.com/download/ollama-linux-amd64-rocm.tgz -o ollama-linux-amd64-rocm.tgz`

`tar xzf ollama-linux-amd64-rocm.tgz`

Copy files from the `ollama` directory into destination directory.

`cp -r lib/ollama/* ~/.local/lib/ollama`  


### Run

`salloc -p gpu-dev --gres=gpu:8 --account=pawsey0001-gpu`

`ollama serve &`

`ollama run deepseek-r1:671b --verbose`

**IMPORTANT**: the models are by default stored under `$HOME/.ollama` which will result
in exceeding the quota when downloading models.
You can either create a symbolic link to a path on a separate filesystem or specify the
model path trough the OLLAMA_MODELS environment variable.

## 3. Install llama.cpp

llama.cpp is the only inference engine that supports all the accelerator
architectures including Vulkan which allows to mix GPUs and CPUs from different
vendors.

Docker containers exist but it's normally better to build from source to customise
the configuration according to your needs.

### AMD MI GPUs

1. Load the relevant modules, in my case:

```
module load gcc/12.2.0
module load rocm/5.7.3
```

2. Set the path to the C and  C++ compilers

```
export CC=/opt/cray/pe/gcc/12.2.0/snos/bin/gcc
export CC++=/opt/cray/pe/gcc/12.2.0/snos/bin/g++
```

3. Invoke cmake

```
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
    cmake -S . -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx90a -DCMAKE_INSTALL_PREFIX=~/.local -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build  --config Release -- -j 16
```

`gfx90a` matches the MI 250x architecture, for a list of all the supported
architectures consult this page: https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html  

4. Install

`cmake --install ./build`

5. Run

Allocate GPU node:

`salloc -p gpu-dev --gres=gpu:8 --account=pawsey0001-gpu`

Run model:

llama-cli -hf MaziyarPanahi/Meta-Llama-3.1-70B-Instruct-GGUF -ngl 9999`

Run model:

`llama-cli -hf MaziyarPanahi/Meta-Llama-3.1-70B-Instruct-GGUF -ngl 9999`

`-ngl 999`: simply tries to upload as many layers as possible on the GPU.

`-hf`: download the model from Huggingface.

Here is the output showing the use of 8 GPUs.

Each of the four GPUs on the node is seen as two GPUs when used from CUDA
or HIP, but multiple GPUs are seen as a single logical one when
accessed through Vulkan and OpenGL.

output:

```
ggml_cuda_init: found 8 ROCm devices:
  Device 0: AMD Instinct MI250X, gfx90a:sramecc+:xnack- (0x90a), VMM: no, Wave Size: 64
  Device 1: AMD Instinct MI250X, gfx90a:sramecc+:xnack- (0x90a), VMM: no, Wave Size: 64
  Device 2: AMD Instinct MI250X, gfx90a:sramecc+:xnack- (0x90a), VMM: no, Wave Size: 64
  Device 3: AMD Instinct MI250X, gfx90a:sramecc+:xnack- (0x90a), VMM: no, Wave Size: 64
  Device 4: AMD Instinct MI250X, gfx90a:sramecc+:xnack- (0x90a), VMM: no, Wave Size: 64
  Device 5: AMD Instinct MI250X, gfx90a:sramecc+:xnack- (0x90a), VMM: no, Wave Size: 64
  Device 6: AMD Instinct MI250X, gfx90a:sramecc+:xnack- (0x90a), VMM: no, Wave Size: 64
  Device 7: AMD Instinct MI250X, gfx90a:sramecc+:xnack- (0x90a), VMM: no, Wave Size: 64
```

*IMPORTANT*: `llama.cpp` stores models inside `$HOME/.cache` it is therefore required
to link `$HOME/.cache` to a directory in a separate filesystem to avoid exceeding
the quota.


## WARNING: Incompatibilities between llama.cpp and ollama

At the time of this writing (April 2025), the GPU version of `ollama` is incompatible 
with the GPU version of `llama.cpp` and therefore the `llama.cpp` files must not be 
in `PATH` and `LD_LIBRARY_PATH` when running `ollama`.
It is better to install `llama.cpp` in it's own directory and add paths to 
the environment variables when needed.


## Ollama - memory usage 

This is the memory used to store the current biggest open model DeepSeek-r1 with
671 billion parameters on the GPUs
The model runs  at about 15 tokens/s for text and 10 tokens/s
for code on the GPU node.

```
load_tensors: loading model tensors, this can take a while... (mmap = false)
load_tensors: offloading 61 repeating layers to GPU
load_tensors: offloading output layer to GPU
load_tensors: offloaded 62/62 layers to GPU
load_tensors:        ROCm0 model buffer size = 35642.36 MiB
load_tensors:        ROCm1 model buffer size = 52215.30 MiB
load_tensors:        ROCm2 model buffer size = 51287.70 MiB
load_tensors:        ROCm3 model buffer size = 52215.30 MiB
load_tensors:        ROCm4 model buffer size = 52215.30 MiB
load_tensors:        ROCm5 model buffer size = 51287.70 MiB
load_tensors:        ROCm6 model buffer size = 46963.85 MiB
load_tensors:        ROCm7 model buffer size = 43364.99 MiB
load_tensors:          CPU model buffer size =   497.11 MiB
⠧ llama_init_from_model: n_seq_max     = 4
llama_init_from_model: n_ctx         = 8192
llama_init_from_model: n_ctx_per_seq = 2048
llama_init_from_model: n_batch       = 2048
llama_init_from_model: n_ubatch      = 512
llama_init_from_model: flash_attn    = 0
llama_init_from_model: freq_base     = 10000.0
llama_init_from_model: freq_scale    = 0.025
llama_init_from_model: n_ctx_per_seq (2048) < n_ctx_train (163840) -- the full capacity of the model will not be utilized
llama_kv_cache_init: kv_size = 8192, offload = 1, type_k = 'f16', type_v = 'f16', n_layer = 61, can_shift = 0
llama_kv_cache_init:      ROCm0 KV buffer size =  5120.00 MiB
llama_kv_cache_init:      ROCm1 KV buffer size =  5120.00 MiB
llama_kv_cache_init:      ROCm2 KV buffer size =  5120.00 MiB
llama_kv_cache_init:      ROCm3 KV buffer size =  5120.00 MiB
llama_kv_cache_init:      ROCm4 KV buffer size =  5120.00 MiB
llama_kv_cache_init:      ROCm5 KV buffer size =  5120.00 MiB
llama_kv_cache_init:      ROCm6 KV buffer size =  4480.00 MiB
llama_kv_cache_init:      ROCm7 KV buffer size =  3840.00 MiB
llama_init_from_model: KV self size  = 39040.00 MiB, K (f16): 23424.00 MiB, V (f16): 15616.00 MiB
llama_init_from_model:  ROCm_Host  output buffer size =     2.08 MiB
llama_init_from_model: pipeline parallelism enabled (n_copies=4)
⠇ llama_init_from_model:      ROCm0 compute buffer size =  2322.01 MiB
llama_init_from_model:      ROCm1 compute buffer size =  2322.01 MiB
llama_init_from_model:      ROCm2 compute buffer size =  2322.01 MiB
llama_init_from_model:      ROCm3 compute buffer size =  2322.01 MiB
llama_init_from_model:      ROCm4 compute buffer size =  2322.01 MiB
llama_init_from_model:      ROCm5 compute buffer size =  2322.01 MiB
llama_init_from_model:      ROCm6 compute buffer size =  2322.01 MiB
llama_init_from_model:      ROCm7 compute buffer size =  2322.02 MiB
llama_init_from_model:  ROCm_Host compute buffer size =    78.02 MiB
```

