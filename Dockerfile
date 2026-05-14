# syntax=docker/dockerfile:1.7

FROM --platform=linux/amd64 alpine:3.20 AS references
WORKDIR /work
RUN apk add --no-cache ca-certificates curl \
    && curl -fsSL -o /work/references.json.gz \
       https://raw.githubusercontent.com/zanfranceschi/rinha-de-backend-2026/main/resources/references.json.gz

FROM --platform=linux/amd64 gcc:16 AS builder
WORKDIR /src
RUN apt-get update \
    && apt-get install -y --no-install-recommends zlib1g-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY src ./src
RUN mkdir -p /out /index \
    && g++ -O3 -DNDEBUG -std=c++20 -march=haswell -mtune=haswell -mavx2 -mfma -flto -fopenmp src/build_index.cpp -lz -o /out/build-index \
    && g++ -Ofast -DNDEBUG -std=c++20 -march=haswell -mtune=haswell -mavx2 -mfma -flto -fno-exceptions -fno-rtti -pthread -static-libstdc++ -static-libgcc src/server.cpp -o /out/server \
    && gcc -O3 -DNDEBUG -std=c11 -march=haswell -mtune=haswell -flto -static -s src/fd_lb.c -o /out/fd-lb
COPY --from=references /work/references.json.gz /tmp/references.json.gz
RUN /out/build-index /tmp/references.json.gz /index/index.bin 1280 65536 6 \
    && ls -lh /index/index.bin

FROM --platform=linux/amd64 debian:trixie-slim AS runtime
LABEL org.opencontainers.image.title="silent-index"
COPY --from=builder /out/server /server
COPY --from=builder /out/fd-lb /fd-lb
COPY --from=builder /index /index
ENTRYPOINT ["/server"]
