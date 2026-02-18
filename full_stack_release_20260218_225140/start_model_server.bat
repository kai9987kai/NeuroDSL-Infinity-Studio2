@echo off
cd /d %~dp0
model_server\NeuroDSL_Champion_v2.exe --mode serve --host 127.0.0.1 --port 8092
