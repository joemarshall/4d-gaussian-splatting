set CUDA_LAUNCH_BLOCKING=1
compute-sanitizer --tool memcheck python trycrash.py