#!/bin/bash

nvidia-smi
pip install -r requirements.txt
python3 main.py -m asso_enco -b 128  -d enwiki8 -g 0 -layer 6 -emb 300 -nhead 4 -s dgx #> energyLM-wiki2.out
