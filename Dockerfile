FROM python:3.8-slim-buster

WORKDIR /wrk

COPY . /wrk

RUN pip install --no-cache-dir -r requirements.txt


CMD ["python", "app"]
