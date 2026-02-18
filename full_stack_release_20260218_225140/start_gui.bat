@echo off
cd /d %~dp0
if exist gui\NeuroDSL_Power_Champion_GUI_Plus.exe (
  gui\NeuroDSL_Power_Champion_GUI_Plus.exe
) else (
  echo GUI EXE not found in gui\
)
