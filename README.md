## Compile

```sh
mkdir build
cd build
cmake ..
make
```
incase of cmake not finding nvcc do this:

```sh
mkdir build
cd build
CUDACXX=/usr/local/cuda/bin/nvcc cmake ..
make
```

Where /usr/local/cuda/bin/nvcc is a standard path to nvcc