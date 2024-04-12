FROM python:3.9-slim

WORKDIR /wrk

COPY . /wrk

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501


CMD ["python", "app"]
