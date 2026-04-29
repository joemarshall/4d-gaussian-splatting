@echo off
setlocal

if "%~1"=="" exit /b 1
if "%~2"=="" exit /b 1

if not exist "saved_models" mkdir "saved_models"

set "latest="
for /f "delims=" %%F in ('dir /b /a-d /o-d "output\%~1\model_output\*.pth" 2^>nul') do (
    set "latest=%%F"
    goto :found
)

echo No .pth file found in output\%~1\model_output
exit /b 1

:found
copy /y "output\%~1\model_output\%latest%" "saved_models\%~2.pth" >nul
echo Copied "%latest%" to "saved_models\%~2.pth"
exit /b 0