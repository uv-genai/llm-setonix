

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

In order to find the `_posixsubprocess` package you need to make the `lib-download` directory
accessible from `~/.local/lib/python3.13`:

`ln -sf ~/.local/lib64/python3.13/lib-dynload ~/.local/lib/python3.13/lib-dynload`.

## 2. Install Podman

Just download the static Podman archive from here: https://github.com/containers/podman/releases 
Upack and move the executable in the `bin` directory to a directory
in the `PATH` env variable after renaming the executable to `podman`.

## 3. Install Ollama

`curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz`

`tar xzf ollama-linux-amd64.tgz`

Assuming you are installing Ollama under `~/.local`:

`cp bin/ollama ~/.local/bin`
`cp -r lib/ollama ~/.local/lib`

Try it out:

`ollama serve &`



## 4. Install llama.cpp

llama.cpp is the only inference engine that supports all the accelerator
architectures including Vulkan which allows to mix GPUs and CPUs from different
vendors.

Docker containers exist but it's normally better to build from source to customise
the configuration according to your needs.


