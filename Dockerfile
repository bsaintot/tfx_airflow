FROM puckel/docker-airflow:1.10.3

USER root
RUN apt-get update

RUN pip install 'tfx>=0.21.1,<0.22'
RUN pip install 'tensorflow>=2.1,<2.2'
RUN pip install 'tensorboard>=2.1,<2.2"'
RUN pip install Flask==1.0.4
