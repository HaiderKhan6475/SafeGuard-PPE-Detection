@echo off
cd /d "%~dp0backend"
if not exist ".venv" (
  python -m venv .venv
)
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8001 --reload
