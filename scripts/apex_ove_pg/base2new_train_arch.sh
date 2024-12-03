#!/bin/bash

cd ../..

# custom config
DATA=./data
TRAINER=APEX_OVE_PG

DATASET=$1
SEED=$2
GPU=$3

CFG=vit_b16_c4_ep10_batch4+2ctx_vision_1
SHOTS=$4
CTX=$5
EPOCHS=$6
ARCH=$7


DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}_ctx_${CTX}_arch_${ARCH}/${TRAINER}/${CFG}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
else
    CUDA_VISIBLE_DEVICES=$GPU python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base \
    TRAINER.COCOOP.N_CTX ${CTX} \
    OPTIM.MAX_EPOCH ${EPOCHS} \
    MODEL.BACKBONE.NAME ${ARCH}
fi
