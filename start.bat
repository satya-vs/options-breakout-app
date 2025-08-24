@echo off
cd /d %~dp0
python -m venv .venv || goto :venv_done
:venv_done
call .venv\Scripts\activate
pip install -r requirements.txt
if not exist .env copy .env.example .env
echo Edit .env to add your POLYGON_API_KEY
echo Starting server at http://127.0.0.1:8000
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
