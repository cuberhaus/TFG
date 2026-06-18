FROM python:3.14-slim

WORKDIR /app

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

RUN pip install \
        "evidently==0.7.21" \
        "psycopg[binary]==3.2.4" \
        "pandas==2.2.3"

COPY scheduler.py /app/scheduler.py

CMD ["python", "/app/scheduler.py"]
