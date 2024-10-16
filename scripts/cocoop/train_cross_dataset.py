import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpuids", default="0,1,2,3,4,5,6,7", help="GPU ids to train model on")
    parser.add_argument("--epochs", help="Number of training epochs")
    parser.add_argument("--seed", type=int, help="Number of seed")
    args = parser.parse_args()

    for seed in [2,]:
        os.system(f"bash xd_train.sh {seed} {args.epochs} {args.seed}")
