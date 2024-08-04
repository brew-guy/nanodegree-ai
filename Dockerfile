FROM pytorch/pytorch:latest

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install any additional Python packages you need
# RUN pip install --no-cache-dir \
#     numpy \
#     matplotlib \
#     jupyter

# Install requirement from requirements.txt, update pip
COPY requirements.txt /tmp/
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /workspace