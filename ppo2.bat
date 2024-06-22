@echo off
set op_ais=coacAI

REM basesWorkers12x12 basesWorkers16x16noResources TwoBasesBarracks16x16
REM cd C:\Users\dmcat\Desktop\adp_exp\MicroRTS_adprl
REM ppo.bat

for /l %%i in (1,1,10) do (
    echo Loop iteration %%i
    for %%j in (%op_ais%) do (
        echo Running script with parameter %%j
        python microrts_cnn.py --map_name basesWorkers16x16noResources --op_ai %%j
        if errorlevel 1 (
            echo Python script failed with parameter %%j
            exit /b 1
        )
    )
)