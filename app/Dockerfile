FROM ubuntu:16.04

RUN apt-get update -y && \
    apt-get install -y python-pip python-dev nano less


# We copy just the requirements.txt first to leverage Docker cache
COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

COPY ./deployment /app

ENTRYPOINT [ "python" ]

CMD [ "recommendations-app.py" ]
