<div align="center">

# Bayesian Principles Improve Prompt Learning In Vision-Language Models 

<p align="center">
  [<a href="https://arxiv.org/abs/2504.14123"><strong>ArXiv</strong></a>]  
  [<a href="https://mingyukim87.github.io/SynergyNeRF/"><strong>Project</strong></a>] 
  <!-- [<a href="#citation"><strong>BibTeX</strong></a>] -->
</p>

</div>

<!-- <a href="https://arxiv.org/abs/2402.03898"><img src="https://img.shields.io/badge/Paper-arXiv:2402.03898-Green"></a>
<a href=#bibtex><img src="https://img.shields.io/badge/Paper-BibTex-yellow"></a> -->

Official PyTorch implementation of **BP CLIP**, as presented in our paper: \
\
**Bayesian Principles Improve Prompt Learning In Vision-Language Models (AISTATS2025)** \
[Mingyu Kim](https://mingyukim87.github.io/)<sup>1*</sup>, [Jongwoo Ko](https://sites.google.com/view/jongwooko)<sup>2*</sup>, 
and [Mijung Park](https://www.cs.ubc.ca/~mijungp/)<sup>3</sup> \
<sup>1</sup>UBC CS, <sup>2</sup>KAIST AI, <sup>3</sup>UBC CS  
(<sup>*</sup> indicates co-first authors)

<!-- <img src="https://mingyukim87.github.io/SynergyNeRF/img/2_Overview_2.png" width="100%">   -->

## Update
- [x] Training code.
- [x] Inference code.
- [x] Datasets.


## 🚀 How to Run

Official scripts are provided under the following directories:
- `scripts/coop_ove_pg/`
- `scripts/cocoop_ove_pg/`
- `scripts/maple_ove_pg/`
- `scripts/apex_ove_pg/`

> **Note:**  
> Replace `{model}` with one of: `coop`, `cocoop`, `maple`, `apex`.  
> Make sure to set the correct `DATA` path before running any script.  
> Run the commands inside the corresponding script directories.

---

### 🟦 1. Generalization: Base to New Classes

You will need two scripts:
- `base2new_train.sh` &nbsp;&nbsp;→&nbsp; Train on base classes
- `base2new_test.sh` &nbsp;&nbsp;→&nbsp; Evaluate on novel classes

Each script accepts the following arguments:
- `DATASET` &nbsp;&nbsp;*(e.g., `UCF101`, `caltech101`, corresponding to files in `configs/datasets/`)*
- `SEED` &nbsp;&nbsp;*(random seed)*
- `GPUIDS` &nbsp;&nbsp;*(comma-separated GPU id list)*
- `EPOCHS` &nbsp;&nbsp;*(number of training epochs)*
- `LAMBDA1` &nbsp;&nbsp;*(weight for KLD)*

**Example:**

```bash
# Generalization from base to new classes
python scripts/{model}_ove_pg/script_base2new_all_datasets.py --gpuids 0 --epochs 10 
```
---

### 🟩 2. Cross-Dataset Transfer

For cross-dataset transfer experiments, set `lambda_1` to the default value of `0.2`.

You will need two scripts:
- `xd_train.sh` &nbsp;&nbsp;→&nbsp; Train on ImageNet
- `xd_test.sh` &nbsp;&nbsp;→&nbsp; Evaluate on downstream datasets (e.g., `UCF101`, `caltech101`, corresponding to files in `configs/datasets/`)

**Example usage:**

```bash
# Training ImageNet dataset and Evaluate downstreamed datset
python scripts/{model}_ove_pg/script_cross_all_datasets.py --gpuids 0 --epochs 10

