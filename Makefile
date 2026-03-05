.PHONY: install run test docker-build docker-up

install:
	python -m pip install -r requirements.txt

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest -q

docker-build:
	docker build -t inboxpilot-lite .

docker-up:
	docker compose up --build

