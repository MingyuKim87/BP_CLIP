#!/bin/bash

cd ../..

# custom config
DATA=./data
TRAINER=MaPLe_OVE_PG

DATASET=imagenet
SEED=$1
GPUIDS=$2
EPOCHS=$3

CFG=vit_b16_c4_ep10_batch4_ctxv1
SHOTS=16



DIR=output/base2new/train_base/${DATASET}/mcmc_${L}_epochs_${EPOCHS}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
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
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base \
    OPTIM.MAX_EPOCH ${EPOCHS} \
fi
