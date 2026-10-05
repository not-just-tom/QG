# Reproducible environment

`environment.yml` pins the project and notebook dependencies to the versions
recorded in the previous environment export. It replaces a machine-specific
Anaconda base-environment export, which included unrelated applications and a
local install path.

## Create the environment

Install Conda (Miniforge is a good lightweight choice), then from the repository
root run:

```powershell
conda env create --file environment.yml
conda activate qg
```

Select the `qg` Python interpreter/kernel in VS Code for notebooks. On another
device, clone or pull this repository and run the same commands.

The file pins the direct project dependencies. Conda may resolve their
transitive dependencies differently over time. Full exports are platform
specific; Windows and Linux may require different package builds or versions.
GPU drivers and CUDA runtimes are also machine-specific.