# Building and test xe-addons

xe-addons is based on IPEX DpcppBuildExtension (by wrapping Torch extension).

## Basic Requirements

* Ubuntu 22.04/Windows
* gcc 11.04
* GPU Driver
* oneAPI 2024.0
* Python libs:
    - setuptools (<=69.5.1)
    - intel_extension_for_pytorch, torch 

## Env setup

### (Option 1): install ipex-llm

```bash
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/cn/
pip install setuptools==69.5.1
```

### (Option 2): install minor libs (ipex etc)

```bash
pip install -f https://developer.intel.com/ipex-whl-stable-xpu intel_extension_for_pytorch==2.1.10+xpu torch==2.1.0a0
# downgrade numpy
pip install numpy==1.26.4
# If encountering g++: error: unrecognized command-line option '-fsycl'
pip install setuptools==69.5.1
```

## Linux build and install

### Source oneAPI env
```bash
source /opt/intel/oneapi/setvars.sh
```

### Build and install
Build
```bash
python setup.py build
```

Install
```bash
python setup.py install
```

Build wheel & install with wheel
```bash
# Build wheel
python setup.py clean --all bdist_wheel --plat-name manylinux2010_x86_64 --python-tag py3
# Install whell
pip install dist/bigdl_core_xe_addons_xxxxxx
```

## Windows build and install

### Modify `cpp_extension.py` for IPEX

Modify `{conda path}/envs/{your conda env}/lib/site-packages/intel_extension_for_pytorch/xpu/cpp_extension.py`. OneAPI path bug.

Existing
```python
            f"{MKLROOT}/lib/intel64/libmkl_sycl.a",
            f"{MKLROOT}/lib/intel64/libmkl_intel_ilp64.a",
            f"{MKLROOT}/lib/intel64/libmkl_sequential.a",
            f"{MKLROOT}/lib/intel64/libmkl_core.a",
```

Change to
```python
            f"{MKLROOT}/lib/libmkl_sycl.a",
            f"{MKLROOT}/lib/libmkl_intel_ilp64.a",
            f"{MKLROOT}/lib/libmkl_sequential.a",
            f"{MKLROOT}/lib/libmkl_core.a",
```

### Build wheel

### Source oneAPI env
```cmd
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
```

### Build wheel

```bash
python setup.py clean --all bdist_wheel
```

## Development (modify and test)

For example, we just want to add a new kernel to `sdp/sdp_xmx.cpp`.

### Modify `setup.py`. 

We can make a few changes:

* Reduce building time by removing most cpp and only keep necessary ones.
* Change install package name (`bigdl-core-xe-addons-test`) and import name (`xe_addons_test`).

**Note that we need to remove or comment out of scopes APIs and functions (sdp_fp8, sdp_causal_vec).**

```python
# bigdl-core-xe-addons-test package name
# xe_addons_test import name
setup(
    name="bigdl-core-xe-addons-test" + suffix_name,
    version=VERSION,
    ext_modules=[
        DPCPPExtension('xe_addons_test', [
            'xpu_addon_ops.cpp',
            'device.cpp',
            'sdp/sdp_xmx.cpp',
            'sdp/api.cpp',
        ],
        extra_compile_args=["-std=c++20"],
        include_dirs=[include_dir])
    ],
    cmdclass={
        'build_ext': DpcppBuildExtension.with_options(use_ninja=False)
    })
```


### Test new kernel (Check coredump or latency)

```python
import time
import math
import torch
import intel_extension_for_pytorch as ipex

# Input shape
bsz = 1
n_head = 32
length = 4096 + 1 + 2 + 3 + 4
head_dim = 128

# Create q k v in [batch, num_head, seq_length, head_dim]
q = torch.randn([bsz, n_head, length, head_dim], dtype=torch.half, device='xpu')
k = torch.randn([bsz, n_head, length, head_dim], dtype=torch.half, device='xpu')
v = torch.randn([bsz, n_head, length, head_dim], dtype=torch.half, device='xpu')
# mask = torch.randn([bsz, n_head, length, length], dtype=torch.half, device='xpu')
mask = None

for i in range(1):
    # import newly built package
    import xe_addons_test
    q3 = q.clone()
    st = time.time()
    for i in range(1):
        # sdp_causal_new in xe_addons_test
        o = xe_addons_test.sdp_causal_new(q, k, v, mask)
    # Sync GPU for timing
    torch.xpu.synchronize()
    et = time.time()
    print(f"New Kernel Latency {(et - st) * 1000}ms")
    print('-' * 80)
```


### Benchmark and compare kernel (with previous kernel)

**Avoid using the same function name (symbol) with existing kernel (e.g., `sdp_causal`), otherwise you will get wrong result due to symbol overwrite.** In this example, we use `sdp_causal_new`.

```python
import time
import math
import torch
import intel_extension_for_pytorch as ipex


# Input shape
bsz = 1
n_head = 32
length = 4096 + 1 + 2 + 3 + 4
head_dim = 128

# Create q k v in [batch, num_head, seq_length, head_dim]
q = torch.randn([bsz, n_head, length, head_dim], dtype=torch.half, device='xpu')
k = torch.randn([bsz, n_head, length, head_dim], dtype=torch.half, device='xpu')
v = torch.randn([bsz, n_head, length, head_dim], dtype=torch.half, device='xpu')
# mask = torch.randn([bsz, n_head, length, length], dtype=torch.half, device='xpu')
mask = None

for i in range(1):
    # import existing package
    import xe_addons
    q2 = q.clone()
    st = time.time()
    for i in range(1):
        # Get output with existing kernel
        o1 = xe_addons.sdp_causal(q, k, v, mask)
    # Sync GPU for timing
    torch.xpu.synchronize()
    et = time.time()
    print(f"Existing kernel Latency {(et - st) * 1000}ms")

    # import newly built libs
    import xe_addons_test
    q3 = q.clone()
    st = time.time()
    for i in range(1):
        #  Get output with new kernel
        o2 = xe_addons_test.sdp_causal_new(q, k, v, mask)
    # Sync GPU for timing
    torch.xpu.synchronize()
    et = time.time()
    print(f"New Kernel Latency {(et - st) * 1000}ms")
    print('-' * 80)

# Compare tensor values in 1e-4 and 1e-3
print(o1.numel())
print((o1 == o2).count_nonzero())
print(torch.isclose(o1, o2, atol=1e-4, rtol=1e-4).count_nonzero())
print(torch.isclose(o1, o2, atol=1e-3, rtol=1e-3).count_nonzero())
```
