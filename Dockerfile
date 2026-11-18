FROM python:3.12-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY partcapture ./partcapture

ENV PARTCAPTURE_DATA=/data
EXPOSE 8000
VOLUME ["/data"]

CMD ["uvicorn", "partcapture.main:app", "--host", "0.0.0.0", "--port", "8000"]
