# MOFA environment

Created with
```
 CONDA_OVERRIDE_CUDA="11.2" mamba create --name serniczek-mofa -c conda-forge gxx_linux-64 gcc_linux-64 python=3.10 cudatoolkit=11.2
 mamba activate serniczek-mofa
 pip install pip-tools
 pip-compile requirements.in
```
