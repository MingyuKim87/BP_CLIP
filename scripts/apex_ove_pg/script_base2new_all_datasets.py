import os
import argparse

datasets = ["caltech101", "dtd", "eurosat", "fgvc_aircraft", "food101", "oxford_flowers", "oxford_pets", "stanford_cars", "sun397", "ucf101"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpuids", default="0", help="GPU ids to train model on")
    parser.add_argument(
        "--epochs", help="Number of training epochs", default=10
    )
    parser.add_argument(
        "--lambda_1", type=float, help="Weight of KLD", default=0.2
    )
    args = parser.parse_args()
    
    for seed in [1, 2, 3]:
        for dataset in datasets:
            os.system(f"bash base2new_train.sh {dataset} {seed} {args.gpuids} {args.epochs} {args.lambda_1}")
            os.system(f"bash base2new_test.sh {dataset} {seed} {args.gpuids} {args.epochs} {args.lambda_1}")
