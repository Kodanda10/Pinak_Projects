# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /code

# Install uv for dependency management
RUN pip install --no-cache-dir uv

# Copy project metadata for dependency installation first (leverages Docker layer caching)
COPY pyproject.toml uv.lock ./
COPY Pinak_Services/memory_service/pyproject.toml Pinak_Services/memory_service/pyproject.toml

# Install runtime dependencies into a managed virtual environment
RUN uv sync --frozen --project Pinak_Services/memory_service --no-dev

# Copy the application source code
COPY Pinak_Services/memory_service/app Pinak_Services/memory_service/app

# Expose the port the app runs on
EXPOSE 8000

# Run the FastAPI application via uv to ensure the managed environment is used
CMD ["uv", "run", "--project", "Pinak_Services/memory_service", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
