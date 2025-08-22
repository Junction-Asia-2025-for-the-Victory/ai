서버 시작
uvicorn main:app --reload

pip freeze > requirements.txt
pip install -r requirements.txt


# Mac/Linux
source venv/Script/activate