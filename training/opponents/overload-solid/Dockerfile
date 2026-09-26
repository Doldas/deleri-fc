FROM python:3.13-alpine
LABEL football-babylon.protocol-version="1.0" football-babylon.team="overload-solid"
WORKDIR /app
COPY models.py strategy.py server.py ./
USER 65532:65532
EXPOSE 8080
ENTRYPOINT ["python", "server.py"]
