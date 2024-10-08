import os


datasets = [
    "caltech101",
    "dtd",
    "eurosat",
    "fgvc_aircraft",
    "food101",
    "oxford_flowers",
    "oxford_pets",
    "stanford_cars",
    "sun397",
    "ucf101",
]
epochs = [20, 10, 60, 10, 20, 40, 20, 40, 10, 20]
seeds = [1]
shots = [16]
ctxs = [1, 2, 4, 8]
GPOUIDS = "0,1,2,3,4,5,6"

for dataset, epoch in zip(datasets, epochs):
    for seed in seeds:
        for shot in shots:
            for ctx in ctxs:
                os.system(
                    f"bash base2new_train_ablation.sh {dataset} {seed} {GPOUIDS} {shot} {ctx} {epoch}"
                )
                os.system(
                    f"bash base2new_test_ablation.sh {dataset} {seed} {GPOUIDS} {shot} {ctx} {epoch}"
                )
