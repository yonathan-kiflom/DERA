FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install --yes --no-install-recommends git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/dera

COPY environment/requirements-cu121.txt environment/requirements-cu121.txt
RUN python -m pip install --no-cache-dir \
    -r environment/requirements-cu121.txt

COPY . .
RUN python -m pip install --no-cache-dir --no-deps .

ENTRYPOINT ["dera"]
CMD ["doctor"]

