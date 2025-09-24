FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY code/deployment/app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

ENV API_URL=http://api:8000

EXPOSE 3000

CMD ["streamlit", "run", "code/deployment/app/app.py", "--server.port=3000", "--server.address=0.0.0.0"]


