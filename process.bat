REM @echo off
setlocal

REM Initial setup
set "input_file="
set "output_dir="

REM Get the parameters
:parse_args
if "%~1"=="" goto check_input
if "%~1"=="-i" (
    set "input_file=%~2"
    shift
    shift
    goto parse_args
)
if "%~1"=="-o" (
    set "output_dir=%~2"
    shift
    shift
    goto parse_args
)
echo Invalid option: %~1
exit /b 1

REM Check the input
:check_input
if "%input_file%"=="" (
    echo Usage: %~nx0 -i ^<input_file^> [-o ^<output_dir^>]
    exit /b 1
)

REM Change rel path to abs path
for %%I in ("%input_file%") do set "input_file=%%~fI"

REM Default output definition
if "%output_dir%"=="" (
    for %%I in ("%input_file%") do (
        set "output_dir=%%~dpnI"
    )
)

REM Get the sample name
for %%I in ("%input_file%") do set "name=%%~nI"

REM Creat output dictionary
if not exist "%output_dir%" (
    mkdir "%output_dir%"
)

REM Creat Step1 ~ Step7 folder
for %%S in (Step1_slide Step2_YOLOX Step3_sc_slide Step4_qc Step5_cut Step6_classify) do (
    mkdir "%output_dir%\%%S"
)

echo Folders created successfully in: %output_dir%

REM Define log output path
set log_file="%output_dir%\log_file.txt"


REM Run Python script
python tools\Step1_slide\svs_slide.py -i "%input_file%" -o "%output_dir%\Step1_slide" >> %log_file% 2>&1

python tools\Step2_YOLOX\YOLOX\tools\demo1.py image -n yolox-x -c tools\Step2_YOLOX\YOLOX\YOLOX_weights\best_ckpt.pth --path "%output_dir%\Step1_slide\%name%" --save_dir "%output_dir%\Step2_YOLOX" --conf 0.3 --nms 0.5 --tsize 1024 --save_result --device gpu >> %log_file% 2>&1

python tools\Step3_sc_slide\sc_slide.py "%output_dir%\Step2_YOLOX\%name%"  "%output_dir%\Step1_slide\%name%"  "%output_dir%\Step3_sc_slide" >> %log_file% 2>&1

python tools\Step4_qc\QC.py --test_dir "%output_dir%\Step3_sc_slide\%name%"  --save_dir "%output_dir%\Step4_qc\%name%" >> %log_file% 2>&1

python tools\Step5_cut\Pytorch-UNet-master\predict.py -i "%output_dir%\Step4_qc\%name%" -o "%output_dir%\Step5_cut" >> %log_file% 2>&1

python tools\Step6_classify\efficient_classify.py --test_dir "%output_dir%\Step5_cut\%name%\json_cut_out"   --save_dir "%output_dir%\Step6_classify\%name%" --ori_img_dir "%output_dir%\Step4_qc\%name%" >> %log_file% 2>&1

REM the end
