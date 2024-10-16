#!/bin/bash

# Define directories
# dirs=("coop" "coop_ove" "coop_ove_pg" "cocoop")
# dirs=("coop_ove" "coop_ove_pg" "cocoop")
# dirs=("coop_ove_pg" "cocoop")
# dirs=("cocoop_ove_pg" "cocoop_ove_pg_32")
# dirs=("cocoop_ove_pg_32")
dirs=("cocoop")
lambdas=("0.05" "0.001")
seed="2"

# Iterate over each directory and run the required commands
for dir in "${dirs[@]}"; do
    for lambda in "${lambdas[@]}"; do
        if [[ $dir == *"32"* ]]; then
            new_dirs="cocoop_ove_pg"

            echo "Running in directory: $new_dirs"

            # Navigate to the directory
            cd $new_dirs

            # Show the current directory
            pwd
            
            # Run the second command: eval_cross_all_dasets.py
            CUDA_VISIBLE_DEVICES=1 python eval_cross_all_datasets_fp32.py --gpuids=1 --epoch=5 --lambda1=$lambda
            echo "Completed eval_cross_all_datasets_fp32.py in $new_dirs"

            # Run the first command: eval_domain_all_dasets.py
            CUDA_VISIBLE_DEVICES=1 python eval_domain_all_datasets_fp32.py --gpuids=1 --epoch=5 --lambda1=$lambda
            echo "Completed eval_domain_all_datasets_fp32.py in $new_dirs"
        else
            echo "Running in directory: $dirs"
            
            # Navigate to the directory
            cd $dirs

            # Show the current directory
            pwd
            
            # # Run the second command: eval_cross_all_dasets.py
            # CUDA_VISIBLE_DEVICES=1 python eval_cross_all_datasets.py --seed=$seed --gpuids=1 --epoch=5 --lambda1=$lambda
            # echo "Completed eval_cross_all_datasets.py in $dir"

            # # Run the first command: eval_domain_all_dasets.py
            # CUDA_VISIBLE_DEVICES=1 python eval_domain_all_datasets.py -seed=$seed --gpuids=1 --epoch=5 --lambda1=$lambda
            # echo "Completed eval_domain_all_datasets.py in $dir"

            # Run the second command: eval_cross_all_dasets.py
            CUDA_VISIBLE_DEVICES=2 python eval_cross_all_datasets.py --seed=$seed --gpuids=2 --epoch=5
            echo "Completed eval_cross_all_datasets.py in $dir"

            # # Run the first command: eval_domain_all_dasets.py
            # CUDA_VISIBLE_DEVICES=0 python eval_domain_all_datasets.py -seed=$seed --gpuids=0 --epoch=5
            # echo "Completed eval_domain_all_datasets.py in $dir"
        fi

        # Navigate back to the parent directory
        cd ..
    done    
done

echo "All scripts executed successfully."