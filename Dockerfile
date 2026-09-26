FROM python:3.13-alpine
LABEL football-babylon.protocol-version="1.0" football-babylon.team="my-team-fc"
WORKDIR /app
COPY models.py strategy.py custom_strategy.py server.py team.json tactics.json ./
COPY artifacts/policies/distilled_policy.json ./artifacts/policies/distilled_policy.json
COPY src ./src
USER 65532:65532
EXPOSE 8080
ENTRYPOINT ["python", "server.py"]
