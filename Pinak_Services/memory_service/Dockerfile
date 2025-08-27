
# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /code

# Copy the requirements file into the container
COPY ./requirements.txt /code/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt \
    && useradd --no-log-init -r -m -d /home/app -s /usr/sbin/nologin app \
    && mkdir -p /code/data /code/app \
    && chown -R app:app /code /home/app

# Copy the application code into the container
COPY ./app /code/app
USER app

# Expose the port the app runs on
EXPOSE 8000

# Define the command to run the application
# We use --host 0.0.0.0 to make it accessible from outside the container
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
