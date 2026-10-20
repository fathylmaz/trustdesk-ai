FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md /app/
COPY src /app/src
RUN pip install --no-cache-dir .
EXPOSE 8000
# Mount a locally generated, trusted artifacts/ folder at /app/artifacts:ro.
CMD ["uvicorn", "trustdesk.api:app", "--host", "0.0.0.0", "--port", "8000"]
