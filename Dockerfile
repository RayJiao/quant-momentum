FROM python:3.11-slim

WORKDIR /app

# Install system deps needed by lxml and matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libxml2-dev \
        libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# results/ and data/ are mounted as volumes at runtime
RUN mkdir -p data results

CMD ["python", "run_comparison.py"]
