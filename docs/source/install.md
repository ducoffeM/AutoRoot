# Installation


## Prerequisites

### Python 3.9+ environment

The use of a virtual environment is recommended, and you will need to ensure that the environment use a Python version
greater than 3.9.
This can be achieved for instance either by using [conda](https://docs.conda.io/en/latest/) or by using [pyenv](https://github.com/pyenv/pyenv) (or [pyenv-win](https://github.com/pyenv-win/pyenv-win) on windows)
and [venv](https://docs.python.org/fr/3/library/venv.html) module.

The following examples show how to create a virtual environment with Python version 3.10.13 with the mentioned methods.

#### With conda (all platforms)

```shell
conda create -n do-env python=3.10.13
conda activate do-env
```

# 🚀 Installation
You can install AutoRoot using `pip`:
> [!IMPORTANT]
> 🚨 There is another package named `autoroot` on PyPI. Please ensure you install **`autoroot_torch`** to get this library.


```python
pip install autoroot_torch
```

Alternatively, to install from source:

```python
git clone https://github.com/your-repo/autoroot.git # Replace with your actual repo URL
cd autoroot
pip install .
```

### Pytorch

`autoroot` relies on [Pytorch](https://pytorch.org/).

## Issues

If you have any issue when installing, you may need to update pip and setuptools:

```shell
pip install --upgrade pip setuptools
```

If still not working, please [submit an issue on github](https://github.com/Pruneeuh/AutoRoot/issues).
