#!/bin/bash

nvidia-smi
pip install -r requirements.txt
python3 main.py -m energy -b 64  -d wiki103 -g 0 -layer 6 -emb 300 -nhead 4 -c norm_attn -s dgx #> energyLM-wiki2.out
