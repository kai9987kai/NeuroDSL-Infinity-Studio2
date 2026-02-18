@echo off
cd /d %~dp0
start "Model Server" cmd /k start_model_server.bat
timeout /t 2 >nul
start "Web Console" cmd /k start_web_console.bat
echo Web stack launched. Open http://127.0.0.1:8787
