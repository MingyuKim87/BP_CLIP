import os
import argparse

# datasets = ["caltech101", "dtd", "eurosat", "fgvc_aircraft", "food101", "oxford_flowers", "oxford_pets", "stanford_cars", "sun397", "ucf101"]
datasets = ["fgvc_aircraft", "eurosat", "ucf101", "stanford_cars", "oxford_flowers", "dtd"]
datasets_3 = ["caltech101", "food101", "oxford_pets", "sun397", "ucf101"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpuids", default="0", help="GPU ids to train model on")
    parser.add_argument(
        "--l", help="Number of monte carlo samples", default=10
    )
    parser.add_argument(
        "--epochs", help="Number of training epochs", default=10
    )
    args = parser.parse_args()
    
    for seed in [1, 2, 3]:
        for dataset in datasets:
            # os.system(f"bash base2new_train.sh {dataset} {seed} {args.gpuids} {args.l} {args.epochs}")
            os.system(f"bash base2new_test.sh {dataset} {seed} {args.gpuids} {args.l} {args.epochs}")
