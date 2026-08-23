.PHONY: setup demo train version evaluate test api dashboard docker all

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

demo:
	python scripts/generate_demo_data.py

train:
	python scripts/train.py

version:
	python scripts/version_data.py

evaluate:
	python scripts/evaluate.py

test:
	pytest

all: demo train version test

api:
	uvicorn app.api.main:app --reload

dashboard:
	streamlit run app/dashboard/app.py

docker:
	docker compose up --build
