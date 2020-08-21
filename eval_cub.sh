#!/bin/bash

# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

python evaluate_cub.py --crf --save-viz --dataset cub --restore-from umr/model_60000.pth --save-dir results/cub/ITER_60000/train/ --split train --num-parts 4
