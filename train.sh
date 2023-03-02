#!/usr/bin/env bash

# train from scratch

python train.py gpus='0' backbone="CSPDarknet-s" num_epochs=300 exp_id="coco_CSPDarknet-s_640x640" use_amp=True data_num_workers=4 batch_size=30
