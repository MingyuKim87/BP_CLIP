import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpuids", type=int, default="0,1,2,3,4,5,6,7", help="GPU ids to train model on")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--lambda1", type=float, default=0.2, help="Number of training epochs")
    args = parser.parse_args()

    for seed in [1,]:
        os.system(f"bash xd_train.sh {seed} {args.epochs} {args.lambda1}")
