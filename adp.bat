@echo off
set op_ais=coacAI rojo mayari

for /l %%i in (1,1,10) do (
    echo Loop iteration %%i
    for %%j in (%op_ais%) do (
        echo Running script with parameter %%j
        python microrts_cnn_adp.py --map_name basesWorkers12x12 --base_ai coacAI --op_ai %%j
        if errorlevel 1 (
            echo Python script failed with parameter %%j
            exit /b 1
        )
    )
)
