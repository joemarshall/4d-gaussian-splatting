set PYTORCH_ALLOC_CONF=per_process_memory_fraction:1.0
rem set PYTORCH_ALLOC_CONF=per_process_memory_fraction:1.0,backend:cudaMallocAsync 
rem set PYTORCH_ALLOC_CONF=backend:cudaMallocAsync
rem set PYTORCH_ALLOC_CONF=per_process_memory_fraction:0.9,expandable_segments:True
python train.py --config output\9moving\config.yaml --start_checkpoint auto_latest
