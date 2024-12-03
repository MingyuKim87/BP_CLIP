import os
import argparse

# datasets = ["caltech101", "dtd", "eurosat", "fgvc_aircraft", "food101", "oxford_flowers", "oxford_pets", "stanford_cars", "sun397", "ucf101"]
datasets = ["fgvc_aircraft", "eurosat", "ucf101", "stanford_cars", "oxford_flowers", "dtd"]
datasets_3 = ["caltech101", "food101", "oxford_pets", "sun397", "ucf101"]

lambdas=[0.5, 0.6, 0.7]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpuids", default="0", help="GPU ids to train model on")
    parser.add_argument(
        "--epochs", type=int, help="Number of training epochs", default=10
    )
    args = parser.parse_args()
    
    for seed in [1,]:
        for lambda_1 in lambdas:
            for dataset in datasets:
                os.system(f"bash base2new_train_fp32.sh {dataset} {seed} {args.gpuids} {args.epochs} {lambda_1}")
                os.system(f"bash base2new_test_fp32.sh {dataset} {seed} {args.gpuids} {args.epochs} {lambda_1}")
