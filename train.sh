#!/bin/bash

#SBATCH -J "train with output folder $1 run name:$2" 
#SBATCH -c4 --mem=16G -G1
#SBATCH -p amp16

echo "Starting training run"

#export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"
#export CUDA_HOME=/usr/local/cuda-13.0
#export CPATH=:/usr/local/cuda-13/targets/x86_64-linux/include/


if [ -z "$1" ] || [ -z "$2" ]; then
  echo Usage: train.sh output_folder run_name
  exit -1
fi

export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib

out_path=output/$1/runs/$2
mkdir -p ${out_path}
out_path=$(realpath ${out_path})
cp output/$1/config.yaml ${out_path}/config.yaml
escaped_path=$(printf '%s' "${out_path}" | sed 's/[\\&|]/\\&/g')
sed -i -E "s|model_path:\s*\".*\"|model_path: \"${escaped_path}\"|" ${out_path}/config.yaml
cat ${out_path}/config.yaml
echo "Running train.py on ${out_path}/config.yaml"
python -u train.py --config ${out_path}/config.yaml --start_checkpoint auto_latest    --logpath logs
echo "DONE!"
