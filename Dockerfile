FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY static/ static/
COPY data/ data/

ENV PORT=8000
EXPOSE 8000

# shell form so ${PORT} expands (Koyeb injects PORT)
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
