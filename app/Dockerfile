FROM ubuntu:16.04

RUN apt-get update -y && \
    apt-get install -y python-pip python-dev nano less

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

COPY ./deployment /app

ENTRYPOINT [ "python" ]

CMD [ "recommendations-app.py" ]
