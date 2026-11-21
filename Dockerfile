FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY django_app ./django_app
RUN python -m pip install --no-cache-dir '.[dev]'
EXPOSE 8000
CMD ["uvicorn", "project35.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
