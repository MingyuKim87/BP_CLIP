#!/bin/bash

cd ../..

# custom config
DATA=./data
TRAINER=VPT_OVE

DATASET=$1
SEED=$2
L=$3
EPOCHS=$4
LOADEP=$4
GPUIDS=$5
LAMBDA1=$6

CFG=vit_b16_c4_ep10_batch1_ctxv1
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
    --model-dir output/imagenet/mcmc_${L}_epochs_${EPOCHS}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED} \
    --load-epoch ${LOADEP} \
    --eval-only \
    OPTIM.MAX_EPOCH ${EPOCHS} \
    TRAINER.VPT_OVE.L ${L} \
    TRAINER.VPT_OVE.LAMBDA_1 ${LAMBDA1} 
fi
