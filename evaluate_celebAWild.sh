#!/bin/bash

# Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

DATASET="CelebAWild"
DATA_DIR="data/CelebA"
MODEL_DIR="snapshots_CelebA"
TRAIN_LIST="data/CelebA/MAFL/training.txt"
TEST_LIST="data/CelebA/MAFL/testing.txt"
MODEL="DeepLab50_2branch"

nvidia-smi
read -e -p "Which GPU to use? : " -i "0" GPU

read -e -p "Model: "                    -i $MODEL MODEL

read -e -p "Evaluate on train set and calculate landmark errors? (y/n) : " -i "n" EVAL_TRAIN

read -e -p "Model dir: "                -i $MODEL_DIR MODEL_DIR
ls $MODEL_DIR
echo ""
read -e -p "Model name(s): "             -i "SCOPS_K8"  METHOD

ls $MODEL_DIR/$METHOD/model*
echo ""
read -e -p "Iter(s): "                     -i "100000" ITER


read -e -p "Save Dir: "                 -i "results_CelebA" SAVE_DIR

read -e -p "extra args for test set : " -i "--crf --save-viz" ARG_TEST
read -e -p "extra args for train set : " ARG_TRAIN


for iter in $ITER
do

    for method in $METHOD
    do
        K_index=${method##*_K}
        K_index=${K_index%%_*}

        NUM_PARTS=$((K_index))
        echo 'num classes' $NUM_CLASS

        SNAPSHOT="${MODEL_DIR}/$method/model_${iter}.pth"
        # Testing
        CMD_TEST="CUDA_VISIBLE_DEVICES=${GPU} python evaluate_celebAWild.py $ARG_TEST --dataset $DATASET --data-dir $DATA_DIR --data-list $TEST_LIST --num-parts $NUM_PARTS --model $MODEL --restore-from $SNAPSHOT --save-dir $SAVE_DIR/$method/ITER_$iter/test/"

        echo ""
        echo "$CMD_TEST"
        echo ""
        eval "$CMD_TEST"


        if [[ $EVAL_TRAIN == "y" ]]
        then
            if [ ! -d "$SAVE_DIR/$method/ITER_$iter/train/" ]; then
                CMD_TRAIN="CUDA_VISIBLE_DEVICES=${GPU} python evaluate_celebAWild.py $ARG_TRAIN --dataset $DATASET --data-dir $DATA_DIR --data-list $TRAIN_LIST --num-parts $NUM_PARTS --model $MODEL --restore-from $SNAPSHOT --save-dir $SAVE_DIR/$method/ITER_$iter/train/"

                echo ""
                echo "$CMD_TRAIN"
                echo ""
                eval "$CMD_TRAIN"
            fi

            echo "Evaluating landmarks"

            CMD_EVAL="python evaluation/face_evaluation_wild.py $SAVE_DIR/$method/ITER_$iter | tee $SAVE_DIR/$method/ITER_$iter/lm_evaluation.txt"
        fi

        echo ""
        echo "$CMD_EVAL"
        echo ""
        eval "$CMD_EVAL"

        CMD_WEB="python web_visualize.py -o $SAVE_DIR/$method/ITER_$iter/web_html -dirs ../datasets/CelebA/img_celeba $SAVE_DIR/$method/ITER_$iter/test/landmarks $SAVE_DIR/$method/ITER_$iter/test/part_overlay $SAVE_DIR/$method/ITER_$iter/test/part_dcrf_overlay $SAVE_DIR/$method/ITER_$iter/test/part_map $SAVE_DIR/$method/ITER_$iter/test/part_map_dcrf -names Img Landmarks PartoOverlay PartoOverlayDCRF PartMaps PartMapsDCRF -ref 1"

        echo ""
        echo "$CMD_WEB"
        echo ""
        eval "$CMD_WEB"
    done
done
