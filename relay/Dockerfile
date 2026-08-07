# syntax=docker/dockerfile:1
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# The .p8 lives here. Do not run as root.
# NB: this is why /keys/apns.p8 must be world-readable on the host, and why
# ./secrets is mounted as a DIRECTORY (a single-FILE bind fails in runc on
# Unraid's /mnt/user shfs: "not a directory").
USER nobody

EXPOSE 8090

# Python stdlib only — this image ships no curl, and adding one for a health
# check would mean a shell + network tooling in the container holding the key.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8090/healthz', timeout=4).status == 200 else 1)"]

# --factory: main.py exposes create_app(transport=None) so tests can inject an
# httpx.MockTransport and run fully offline.
CMD ["uvicorn", "main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8090"]
