# GenAI environment configuation

This document explain how to setup a GenAI development environment
on the GPU nodes of the Setonix Cray-EX supercomputer at the
Pawsey Surpercomputing Centre.
Since all the packages need to be built and installed from scratch
in the home directory the instructions should be generic enough
to apply to any other Linux cluster.

The main component of any GenAI environment is the inference platform
to serve the models.

In this document we are going to cover the installation and usage of
the following serving platforms:
1. ollama
2. llama.cpp

The advanced platforms such as *SGLang* and *vLLM* that should be faster
and allow models to be distributed on multiple nodes are not available
because of issues with the containers environment (Singularity instead of
Podman/Docker), the ROCm version (>= 6.2 required) and the lack of support
for the gfx90a (MI250) platform in the case of *SGLang*.

llama.cpp has recently introduced the ability to perform distributed
inference by re-compiling with a specific clag but I was not able to test it yes 
and the documentaiton convers only CUDA.
See [here](https://github.com/ggml-org/llama.cpp/tree/master/examples/rpc).

Given the fact that the currently biggest model (DeepSeek-r1) can
ran on a single node at a reasonable speed it might not be
required to use SGLang or vLLM to distribute a model on
multiple nodes.

All the platforms expose both a chat and an OpenAI compatible
endpoint.

In general it is not possible to create working GenAI environments
on systems with Singularity because web services need to be run
with additional privileges, not to mention a command line interface
different from podman and docker which makes it hard to use pre-existing
recipes and the lack of health check tools (--healthcheck-*).

SLURM is ok for development but cannot be used to serve production workloads
since the auto-scaling features provided by Kuberneted are required.

For reference when trying to run a web service (e.g. nginx) through
Singularity on Setonix without sudo you get:
`bind() to 0.0.0.0:80 failed (13: Permission denied)`.

One option is to run the model serving service on Setonix and connect
to it from another system through an ssh tunnel, installing all the
required tools elsewhere.

Ollama requires two processes to run: a client and a server but
it's easier to use because it can use its own models dwnloaded
from the ollama site.

llama.cpp is the only platform that works with any acceleration
infrastructure from Vulkan to Metal and SYCL and only requires
a single process running. It does however only support models
in the =gguf= format which requires in most cases downloading
the models from Huggingface and converting it with the
=convert_hf_to_gguf_update= tool.

llama.cpp is in general faster than ollama.

This article shows how to install both ollama and llama.cpp but
only one is required to serve the models.


## 1. Install Python

A recent version of Python is required to guarantee that all
the packages work properly.

At the time of this writing Python 3.13.3 is the latest version but
because some tools like llama.cpp still rely on 3.12 it is better to install
the previous version.

Since on shared systems Python cannot be installed using a package manager
it needs to be built from source code.

Procedure to follow after having downloaded and unpacked
the python source archive downloaded from https://www.python.org/downloads/source/
to install Python and `pip` under `~/.local`:

1. `module load gcc` (Cray)
2. `cd Python-3.12.9`
3. `./configure --prefix=$HOME/.local -enable-optimizations --with-lto=full --with-ensurepip`
4. `make -j 32`
5. `make install`

In order to find the `_posixsubprocess` package you need to make the `lib64/lib-dynload` directory
accessible from `~/.local/lib/python3.12`:

`ln -sf ~/.local/lib64/python3.12/lib-dynload ~/.local/lib/python3.12/lib-dynload`.


## 2. Install Ollama

Download the version for your system from here: https://github.com/ollama/ollama/blob/main/docs/linux.md

**WARNING**: the models are by default stored under `$HOME/.ollama` which will
result in exceeding the quota when downloading models.
You can either create a symbolic link to a path on a separate filesystem or
specify the model path trough the OLLAMA_MODELS environment variable.

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

After having installed the CPU/NVIDIA version, download the ROCm version:

`curl -L https://ollama.com/download/ollama-linux-amd64-rocm.tgz -o ollama-linux-amd64-rocm.tgz`

`tar xzf ollama-linux-amd64-rocm.tgz`

Copy files from the `ollama` directory into destination directory.

`cp -r lib/ollama/* ~/.local/lib/ollama`


### Run

`salloc --mem=0 -p gpu-dev --gres=gpu:8 --account=pawsey0001-gpu`

`ollama serve &`

`ollama run deepseek-r1:671b --verbose` Use a smaller model for testing.


## 3. Install llama.cpp

llama.cpp is the only inference engine that supports all the accelerator
architectures including Vulkan which allows to mix GPUs and CPUs from different
vendors.

Also while ollama requires two processes to run, the server and the client,
llama.cpp requires only one.

Docker containers exist but it's normally better to build from source to customise
the configuration according to your needs and in any case with Singularity it
won't be possible to access llama.cpp through the REST endpoints.

**WARNING**: `llama.cpp` stores models inside `$HOME/.cache` it is therefore required
to link `$HOME/.cache` to a directory in a separate filesystem to avoid exceeding
the quota.

Start by cloning the git repository:

`git clone https://github.com/ggml-org/llama.cpp.git`

### AMD MI GPUs


**1. Load the relevant modules**:

```
module load cmake/3.27.7
module load gcc/12.2.0
module load rocm/5.7.3
```

**2. Set the path to the C and  C++ compilers**

```
export CC=/opt/cray/pe/gcc/12.2.0/snos/bin/gcc
export CCXX=/opt/cray/pe/gcc/12.2.0/snos/bin/g++
```

**3. Invoke cmake**

Because of incompatibilites with ollama it is better to install
llama.cpp in a directory not in PATH and activate as needed;
to install under `~/.opt/llama.cpp` you can use the following command line

```
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
    cmake -S . -B build-rocm -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx90a -DCMAKE_INSTALL_PREFIX=~/.opt/llama.cpp -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build-rocm --config Release -- -j 32
```

`gfx90a` matches the MI 250x architecture, for a list of all the supported
architectures consult this page: https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html  

**4. Install**

`cmake --install ./build`

**5. Install commmand line tools**

The models read by llama.cpp need to be in =gguf= format which in many cases
requires conversion from the =safetensors= format used by most models
on Huggingface.

To build all the tools =cd= into llama.cpp git repository and run the
following commands:

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements
```

The tools include:

* =convert_hf_to_gguf.py= convert from safetensors Hugginface model to gguf
* =convert_llama_ggml.py=_to_gguf convert from ggml tensors to gguf
* =convert_lora_to_gguf.py= convert LoRA finetuned model to gguf

**6. Run**

Allocate GPU node:

`salloc --mem=0 -p gpu-dev --gres=gpu:8 --account=pawsey0001-gpu`

Run model:

`llama-cli -hf MaziyarPanahi/Meta-Llama-3.1-70B-Instruct-GGUF -ngl 999 -cnv`

`-ngl 999`: simply tries to upload as many layers as possible on the GPU.

`-hf`: download the model from Huggingface.

`-cnv`: conversation (interactive) mode.

If the loading time is too long try to add the `--no-mmap` switch
to the command line, note however tha if the model is too big it won't
fit in memory without =mmap= enabled and loading will fail.

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

##4. Use models from Hugginface with llama.cpp

When using llama.cpp you need to use models from Huggingface and convert the
non-gguf ones (most of them) to the =gguf= format.

Create a Hugginface account first and generate an access token.

Install the huggingface command line:

`pip install -U "huggingface_hub[cli]`

To speed up donwload also install =hf_xet=

`pip install hf_xet`

To download a model from Huggingface:
    1. go to the Huggingface website and navigate to the model you want to download
    2. make sure to accept the terms and condtions for the model which might require
    to wait for permission to use it
    3. log into Huggingface using `huggingface-cli login`
    4. run `huggingface-cli download <model name>`

By default the model is downloaded into the =~/.cache/huggingface= directory,
using a specific format suitable for consumption through the Huggingface API,
to make sure the model is downloaed in the proper format for llama to work
use the command line:

`huggingface-cli download <model name> --local-dir <local dir> --include "*"`

E.g.

`huggingface-cli download google/gemma-3-4b-it --local-dir gemma3-4b-model --include "*"`

Before downloading the model look for =gguf= models on Huggingface, *unsloth* does
normally have the latest ones available.

Convert the model to =gguf=:

`python3 llama.cpp/convert_hf_to_gguf.py gemma3-4b-model --outfile gemma3-4b.gguf`

The model can now be run through llama.cpp:

`llama-cli -m gemma3-4b.gguf -cnv`

Optionally quantise the model to reduce the size using `llama-quantize`.

## Limitations of gfx90a architecture

The AMD MI250 GPUs do not support natively 4 and 6 bit types and therefore
it is not possible to use the quantised models, attempts to do so
will result in the following warning:

`load_tensors: tensor 'token_embd.weight' (q4_K) (and 0 others) cannot be used with preferred buffer type ROCm_Host, using CPU instead`

gfx942 and above is required to make use of the smaller quantised models.


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
## Errors when building SGLang container

The standard ROCm SGLang containers is configured to work
onlyt with gfx942 hardware, trying to switch to gfx90a
results in many errors like the following one:
```
In file included from /sgl-workspace/aiter/aiter/jit/build/ck/include/ck_tile/ops/flatmm/block/flatmm_32x512x128_1x4x1_16x16x32_hip.hpp:461:
/sgl-workspace/aiter/aiter/jit/build/ck/include/ck_tile/ops/flatmm/block/uk/flatmm_uk_gfx9_32x512x128_1x1x1_16x16x16.inc:671:5: error: instruction not supported on this GPU
  671 |     _UK_PIPELINE_0(_UK_GLD_A0, _UK_GLD_A1, _UK_GLD_A2, _UK_GLD_A3, _UK_GLD_A4, _UK_GLD_A5, _UK_GLD_A6, _UK_GLD_A7_AND_L1 ,
      |     ^
```

## Scripts to activate/deactivate llama.cpp

Assuming llama.cpp is installed under =~/.opt/llama.cpp=
you can =source= the following scripts to activate/deactivate it:

Activate:
```
module load rocm
export BEFORE_LLAMA_CPP_PATH=$PATH
export BEFORE_LLAMA_CPP_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
export PATH=~/.opt/llama.cpp/bin:$PATH
export LD_LIBRARY_PATH=~/.opt/llama.cpp/lib64:$LD_LIBRARY_PATH
```

De-activate:
```
export PATH=$BEFORE_LLAMA_CPP_PATH
export LD_LIBRARY_PATH=$BEFORE_LLAMA_CPP_LD_LIBRARY_PATH
module unload rocm
```

## Parsing local llama.cpp file names

When downloading models from Higgingface with the =hf= switch
the files are stored under `~/.cache/llama.cpp`.

When running again llama.cpp to load a cached model you need
to use the same name used when downloading from Huggingface.

The Huggingface GGUF model name format is as follows:

`<provider>/<model name>-GGUF:<quantisation>`

E.g:

`unsloth/gemma-3-4b-it-GGUF:Q8_0`

translated by llama.cpp to

`<provider>_<model name>-GGUF_<model name>-<quantisation>`

E.g:

`unsloth_gemma-3-4b-it-GGUF_gemma-3-4b-it-Q8_0`
