FROM astrocrpublic.azurecr.io/runtime:3.1-4

RUN mkdir -p /usr/local/airflow/data/feature_store/features \
             /usr/local/airflow/data/feature_store/metadata \
             /usr/local/airflow/data/feature_store/transformers \
             /usr/local/airflow/models/transformers \
             /usr/local/airflow/mlruns \
    && chmod -R 777 /usr/local/airflow/data \
    && chmod -R 777 /usr/local/airflow/models \
    && chmod -R 777 /usr/local/airflow/mlruns

USER root

RUN mkdir -p /mlflow-artifacts \
    && chmod -R 777 /mlflow-artifacts