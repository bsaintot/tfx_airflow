FROM puckel/docker-airflow:1.10.3

USER root
RUN apt-get update

COPY constants.py $AIRFLOW_HOME/dags/constants.py
COPY trainer.py $AIRFLOW_HOME/dags/trainer.py
COPY transform.py $AIRFLOW_HOME/dags/transform.py
COPY pipeline.py $AIRFLOW_HOME/dags/pipeline.py

COPY data $AIRFLOW_HOME/dags/data
COPY setup.py $AIRFLOW_HOME/setup.py

RUN pip install .
