export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"
export CUDA_HOME=/usr/local/cuda-13.0
export CPATH=:/usr/local/cuda-13/targets/x86_64-linux/include/
#conda env create -f environment.yml 
echo Build simple knn
#conda run -n 4dgs python -m pip install ./simple-knn --no-build-isolation
echo Build pointops2
conda run -n 4dgs python -m pip install ./pointops2 --no-build-isolation
