#!/bin/bash

cd ../..

# custom config
DATA=./data
TRAINER=CoCoOp_OVE_PG

DATASET=$1
SEED=$2
EPOCHS=$3
LOADEP=$3
GPUIDS=$4
LAMBDA1=$5

CFG=vit_b16_c4_ep10_batch1_ctxv1_fp32
SHOTS=16


DIR=output/evaluation/${TRAINER}_${LAMBDA1}/${CFG}_${SHOTS}shots/${DATASET}/seed${SEED}
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
    --model-dir output/imagenet/epochs_${EPOCHS}/${TRAINER}_${LAMBDA1}/${CFG}_${SHOTS}shots/seed${SEED} \
    --load-epoch ${LOADEP} \
    --eval-only \
    OPTIM.MAX_EPOCH ${EPOCHS}
fi
