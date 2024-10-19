# FROM ubuntu:jammy
FROM python:3.10-slim

WORKDIR /application
COPY . .

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
CMD ["uvicorn", "application.main:app", "--host", "0.0.0.0", "--port", "8000"]
