@echo off
REM Windows: double-click to start the worker (edit SERVER and TOKEN first).
set SERVER=https://stt.example.uz
set TOKEN=paste-worker-token-here
cd /d %~dp0\..\..
call .venv\Scripts\activate
:loop
python webapp\worker\worker.py --server %SERVER% --token %TOKEN% --name home-4060
echo Worker stopped, restarting in 30s...
timeout /t 30
goto loop
