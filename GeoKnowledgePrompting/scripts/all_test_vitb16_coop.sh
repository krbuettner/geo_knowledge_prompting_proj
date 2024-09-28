#!/bin/bash

cd ..

# Command line info
OLD_DATASET=$1
NEW_DATASET=$2
WEIGHT=$3

# Which classes, which epoch 
SUB=all
LOADEP=100

# Ablation data
CFG=vit_b16_ep100_ctxv1 # rn50_ep100
CTP=end  # class token position (end or middle)
NCTX=4  # number of context tokens
SHOTS=16  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)

# Main config
DATA=path_to_data
TRAINER=CoOp



for SEED in 1 2 3
do
    COMMON_DIR=${OLD_DATASET}/shots_${SHOTS}_${WEIGHT}/${TRAINER}/${CFG}/seed${SEED}
    # EX: geode/shots_8.0/KgCoOp/vit_b16_ep100_ctxv1/seed1
    MODEL_DIR=output/all/train_${SUB}/${COMMON_DIR}
    # EX: output/train_all/geode/shots_8.0/KgCoOp/vit_b16_ep100_ctxv1/seed1
    DIR=output/all/test_${SUB}/${NEW_DATASET}/${COMMON_DIR}
    # EX: output/test_all/geode/shots_8.0/KgCoOp/vit_b16_ep100_ctxv1/seed1

    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    else
        echo "Run this job and save the output to ${DIR}"
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${NEW_DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --model-dir ${MODEL_DIR} \
        --load-epoch ${LOADEP} \
        --eval-only \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES ${SUB}
    fi
done
