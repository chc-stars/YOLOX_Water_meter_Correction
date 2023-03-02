#!/usr/bin/env bash

# evaluate customer dataset
python evaluate.py gpus='0' backbone="CSPDarknet-s" load_model="exp/coco_CSPDarknet-s_640x640/model_best.pth" batch_size=24


