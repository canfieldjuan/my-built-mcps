FROM python:3.11-slim

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash sandbox

# Install minimal dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Setup sandbox environment
WORKDIR /sandbox
RUN chown sandbox:sandbox /sandbox

# Switch to non-root user
USER sandbox

# No entrypoint - will be specified at runtime
