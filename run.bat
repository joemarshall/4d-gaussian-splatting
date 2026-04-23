set PYTORCH_ALLOC_CONF=per_process_memory_fraction:1.0
rem set PYTORCH_ALLOC_CONF=per_process_memory_fraction:1.0,backend:cudaMallocAsync 
rem set PYTORCH_ALLOC_CONF=backend:cudaMallocAsync
rem set PYTORCH_ALLOC_CONF=per_process_memory_fraction:0.9,expandable_segments:True
set CUDA_LAUNCH_BLOCKING=0

python train.py --config output\9moving\config.yaml --start_checkpoint auto_latest    --logpath logs
::python train.py --config output\9moving\config.yaml --start_checkpoint auto_latest   --use_fastgs_densification --logpath logs

::python train.py --config output\9moving\config.yaml --start_checkpoint auto_latest  --use_fastgs_densification --use_lff_densification --logpath logs
::python train.py --config output\9moving\config.yaml --start_checkpoint auto_latest   --logpath logs
::compute-sanitizer --tool initcheck python train.py --config output\9moving\config.yaml --start_checkpoint auto_latest --use_fastgs_densification --logpath logs 
::compute-sanitizer --tool memcheck python train.py --config output\9moving\config.yaml --start_checkpoint auto_latest --use_fastgs_densification --logpath logs 
