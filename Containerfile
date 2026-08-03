FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir mcp pydantic uvicorn

COPY server.py /app/server.py

EXPOSE 8000

ENTRYPOINT ["python", "/app/server.py"]
