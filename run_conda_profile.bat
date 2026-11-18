setlocal
setlocal
call %LOCALAPPDATA%\miniconda3\condabin\conda_hook.bat
call conda activate 4d2
python train.py --config output\esper_ball\config.yaml --start_checkpoint auto_latest    --logpath logs
