FROM python:3.11-slim
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1
RUN pip install --upgrade pip
# Install Parlant server
RUN pip install parlant==3.0.1
# Expose default port
EXPOSE 8800
# Provide a default data dir
ENV PARLANT_DATA_DIR=/data
RUN mkdir -p /data
# Run the official server entrypoint
CMD ["parlant-server"]

