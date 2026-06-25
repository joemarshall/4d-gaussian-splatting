#!/bin/bash

#SBATCH -J "Prepare videos for training" 
#SBATCH -c16 --mem=16G -G1
#SBATCH -p amp20

mkdir -p output/$1
python -u train_with_unknown_cameras.py /data/${USER}/4dgs/$1 -o output/$1 --colmap_path $(which colmap)