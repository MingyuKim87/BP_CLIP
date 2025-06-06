#!/bin/bash

cd ../..

# custom config
DATA=./data
TRAINER=CoOp

DATASET=$1
SEED=$2
EPOCHS=$3
LOADEP=$3
GPUIDS=$4

CFG=vit_b16_c4_ep10_batch2_ctxv1
SHOTS=16


DIR=output/evaluation/${TRAINER}/${CFG}_${SHOTS}shots/${DATASET}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
else
    CUDA_VISIBLE_DEVICES=${GPUIDS} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir output/imagenet/epochs_${EPOCHS}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED} \
    --load-epoch ${LOADEP} \
    --eval-only \
    OPTIM.MAX_EPOCH ${EPOCHS} 
fi
