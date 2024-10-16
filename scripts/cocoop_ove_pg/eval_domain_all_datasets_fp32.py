import os
import argparse

# datasets = ["caltech101", "dtd", "eurosat", "fgvc_aircraft", "food101", "oxford_flowers", "oxford_pets", "stanford_cars", "sun397", "ucf101"]
datasets = ["imagenetv2", "imagenet_sketch", "imagenet_a", "imagenet_r"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpuids", type=int, default="0", help="GPU ids to train model on")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--lambda1", type=float, default=0.2, help="Number of training epochs")
    args = parser.parse_args()

    for seed in [1, ]:
        for dataset in datasets:
            os.system(f"bash xd_test_fp32.sh {dataset} {seed} {args.epochs} {args.gpuids} {args.lambda1}")
