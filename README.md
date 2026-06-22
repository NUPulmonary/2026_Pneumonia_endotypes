# Pathogen-specific host responses define distinct pneumonia endotypes in the human lung

This repository will have the code used to produce analysis and figures for the
“[Pathogen-specific host responses define distinct pneumonia endotypes in the human lung](https://www.biorxiv.org/content/10.64898/2026.05.12.724509)”
manuscript.

The website to explore data is available here https://sqlifts.fsm.northwestern.edu/public/pneumonia-endotypes/
and its source code can be found in `08_website`.

## Python environments

We used the **main** environment for everything, except preparing data for MOFA and fittin
the MOFA model. That environment is detailed in [06_mofa](06_mofa) folder.
All the dependencies are listed in [requirements.freeze.txt](requirements.freeze.txt).
`Geneformer` was installed from source from [huggingface](https://huggingface.co/ctheodoris/Geneformer/tree/main)
using commit `39ab62e6cf44d3dc06258870ec21f9d8b98aec0c` (7 Sep 2023). `torch` wheels were
downloaded manually from https://download.pytorch.org/whl/torch/

Environment was created with
```shell
CONDA_OVERRIDE_CUDA="11.2" mamba create --prefix=/projects/b1196/envs/serniczek -c conda-forge gxx_linux-64 gcc_linux-64 python=3.10 cudatoolkit=11.2
```
with python `3.10.12`.
