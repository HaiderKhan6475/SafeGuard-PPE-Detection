@echo off
cd /d "%~dp0frontend"
if not exist ".venv" (
  python -m venv .venv
)
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
