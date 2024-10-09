#!/bin/bash

# Define directories
# dirs=("coop" "coop_ove" "coop_ove_pg" "cocoop")
# dirs=("coop_ove" "coop_ove_pg" "cocoop")
# dirs=("coop_ove_pg" "cocoop")
dirs=("cocoop_ove_pg" "cocoop_ove_pg_32")

# Iterate over each directory and run the required commands
for dir in "${dirs[@]}"; do
    
    
    if [[ $dir == *"32"* ]]; then
        new_dirs="coop_ove_pg"

        echo "Running in directory: $new_dirs"

        # Navigate to the directory
        cd $new_dirs

        # Show the current directory
        pwd
        
        # Run the first command: eval_domain_all_dasets.py
        CUDA_VISIBLE_DEVICES=0 python eval_domain_all_datasets_fp32.py --gpuids=0 --epoch=5
        echo "Completed eval_domain_all_datasets.py in $new_dirs"

        # Run the second command: eval_cross_all_dasets.py
        CUDA_VISIBLE_DEVICES=0 python eval_cross_all_datasets_fp32.py --gpuids=0 --epoch=5
        echo "Completed eval_cross_all_datasets.py in $new_dirs"
    else
        echo "Running in directory: $dirs"
        
        # Navigate to the directory
        cd $dirs

        # Show the current directory
        pwd
        
        # Run the first command: eval_domain_all_dasets.py
        CUDA_VISIBLE_DEVICES=0 python eval_domain_all_datasets.py --gpuids=0 --epoch=5
        echo "Completed eval_domain_all_datasets.py in $dir"

        # Run the second command: eval_cross_all_dasets.py
        CUDA_VISIBLE_DEVICES=0 python eval_cross_all_datasets.py --gpuids=0 --epoch=5
        echo "Completed eval_cross_all_datasets.py in $dir"
    fi

    # Navigate back to the parent directory
    cd ..
done

echo "All scripts executed successfully."