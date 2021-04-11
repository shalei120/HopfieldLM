#!/bin/bash

nvidia-smi
python3 main.py -m energy -b 128  -d wiki2 -g 0 #> energyLM-wiki2.out
