#!/bin/bash

cd ../..

# custom config
DATA=./data
TRAINER=CoOp_OVE_PG

DATASET=imagenet
SEED=$1
EPOCHS=$2

CFG=vit_b16_c4_ep10_batch2_ctxv1_fp32
SHOTS=16


DIR=output/${DATASET}/epochs_${EPOCHS}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}
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
    OPTIM.MAX_EPOCH ${EPOCHS}
fi
