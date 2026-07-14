.PHONY: all data features train score serve dashboard test clean

all: data features train score

data:
	python -m churn_platform.data_generation --n 200000 --out data/raw/customers.parquet

features:
	python -m churn_platform.spark_etl --config config/config.yaml

train:
	python -m churn_platform.train --config config/config.yaml

score:
	python -m churn_platform.score --config config/config.yaml

serve:
	uvicorn churn_platform.serving.app:app --port 8000

dashboard:
	streamlit run dashboards/streamlit_app.py

test:
	pytest tests/ -v

clean:
	rm -rf data/processed models/*.joblib reports/*
