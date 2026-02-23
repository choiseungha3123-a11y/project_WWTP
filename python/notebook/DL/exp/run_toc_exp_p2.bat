@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d c:\project_WWTP\python
"C:\Users\kimuc\miniconda3\envs\AIproject\python.exe" "notebook\DL\exp\toc_experiment.py" --phase 2 --best-hidden 384 --best-layers 1 --best-lr 0.001
