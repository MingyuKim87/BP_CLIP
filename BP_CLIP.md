## How to Run

The running scripts are provided in `scripts/coop_ove_pg/`, `scripts/cocoop_ove_pg/`, `scripts/maple_ove_pg/` and `scripts/apex_ove_pg/` . Make sure you change the path in `DATA` and run the commands under `scripts/coop_ove_pg/`, `scripts/cocoop_ove_pg/`, `scripts/maple_ove_pg/` or `scripts/apex_ove_pg/`.

### Generalization From Base to New Classes

You will need both `scripts/coop_ove_pg/base2new_train_fp32.sh` and `scripts/coop_ove_pg/base2new_test_fp32.sh`. The former trains a model on base classes while the latter evaluates the trained model on new classes. Both scripts have file input arguments, i.e.:
* `DATASET` (takes as input a dataset name, like `imagenet` or `caltech101`. The valid names are the files' names in `configs/datasets/`.)
* `SEED`(Seed number)
* `GPUIDS` (List of gpu ids, should be provided as a sequence of number, separated by ",")
* `EPOCHS` (Number of training epochs)
* `LAMBDA1` (Weight of KLD)

Below we provide an example on how to train and evaluate the model on all datasets.

```bash
# Generalization from base to New classes
python train_eval_bayesian.py --gpuids 0 --epochs 10 

# Generalization from base to New classes using float32
python train_eval_bayesian_fp32.py --gpuids 0 --epochs 10 
```

### Cross-Dataset Transfer

In this experiment, the lambda_1 is set to the default 0.2

```bash
# float16
python train_cross_dataset.py --gpuids 0 --epochs 10
python eval_cross_all_datasets.py --gpuids 0 --epochs 10

# float32
python train_cross_dataset_fp32.py --gpuids 0 --epochs 10
python eval_cross_all_datasets_fp32.py --gpuids 0 --epochs 10
```