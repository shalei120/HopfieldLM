#!/bin/bash

#nvidia-smi
pip install -r requirements.txt
python3 main_mt.py -m transformer -b 64  -d DE_EN -g 0 -layer 6 -emb 512 -s dgx #> energyLM-wiki2.out
