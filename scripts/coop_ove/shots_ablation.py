import os


datasets = ["caltech101", "dtd", "eurosat", "fgvc_aircraft", "food101", "oxford_flowers", "oxford_pets", "stanford_cars", "sun397", "ucf101"]
epochs   = [20,10,60,10,20,40,20,40,10,20] 


seeds   = [1]
shots   = [1,4,8]
GPOUIDS = "0,1,2,3,4,5,6,7"

for dataset, epoch in zip(datasets, epochs):
    for seed in seeds:
        for shot in shots:
            os.system(f"bash base2new_train_ablation.sh {dataset} {seed} {GPOUIDS} {shot} {epoch}")
            os.system(f"bash base2new_test_ablation.sh {dataset} {seed} {GPOUIDS} {shot} {epoch}")
