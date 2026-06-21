@echo off
echo Starting W-SafeRoutes Python Backend...
call .venv\Scripts\activate.bat
set COLORPREDICT3_PORT=8100
python colorpredict3.py
pause
