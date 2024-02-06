FROM python:3.10-slim

WORKDIR /app

ADD . /app

RUN apt-get update \
    && apt-get install -y gcc \
    && pip install poetry \
    && poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
