#!/bin/bash

cd ../..

# custom config
DATA=./data
TRAINER=VPT_OVE_PG

DATASET=imagenet
SEED=$1
L=$2
EPOCHS=$3
LAMBDA1=$4

CFG=vit_b16_c4_ep10_batch1_ctxv1
SHOTS=16


DIR=output/${DATASET}/mcmc_${L}_epochs_${EPOCHS}/${TRAINER}_${LAMBDA1}/${CFG}_${SHOTS}shots/seed${SEED}
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
    TRAINER.VPT_OVE.L ${L} \
    TRAINER.VPT_OVE.LAMBDA_1 ${LAMBDA1} 
fi
