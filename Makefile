install:
	pip install -r requirements.txt

train:
	python src/train.py

test:
	pytest tests/  # Asumiremos un dir tests/ con test_train.py

lint:
	black --check src/