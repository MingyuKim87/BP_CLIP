#!/bin/bash

cd ../..

# custom config
DATA=./data
TRAINER=CoCoOp_OVE_PG

DATASET=imagenet
SEED=$1
EPOCHS=$2
LAMBDA1=$3

CFG=vit_b16_c4_ep10_batch1_ctxv1
SHOTS=16


DIR=output/${DATASET}/epochs_${EPOCHS}/${TRAINER}_${LAMBDA1}/${CFG}_${SHOTS}shots/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
else
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    OPTIM.MAX_EPOCH ${EPOCHS} \
    TRAINER.COCOOP_OVE_PG.LAMBDA_1 ${LAMBDA1}
fi
