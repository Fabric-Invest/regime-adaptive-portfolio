FROM public.ecr.aws/amazonlinux/amazonlinux:2023

# Install Python and shadow-utils (for useradd)
RUN dnf install -y python3 python3-pip shadow-utils && \
    dnf clean all && \
    rm -rf /var/cache/dnf

# Create symlink for python command
RUN ln -sf python3 /usr/bin/python

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LOG_LEVEL=INFO

WORKDIR /app

# Create a non-root user and logs directory
RUN useradd -r -s /sbin/nologin appuser && \
    mkdir -p /app/logs && \
    chown -R appuser:appuser /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy strategy code
COPY . .

# Ensure appuser owns all files
RUN chown -R appuser:appuser /app

USER appuser

CMD ["python", "main.py"]

