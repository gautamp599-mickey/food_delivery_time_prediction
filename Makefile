.PHONY: install train run test

install:
	pip install -r requirements.txt

train:
	python -m src.models.train

run:
	streamlit run src/app.py

test:
	pytest tests/ -v