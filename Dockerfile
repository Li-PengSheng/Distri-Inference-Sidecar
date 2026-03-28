FROM rust:1.85-slim AS rust-builder

WORKDIR /app/rust_ops
COPY rust_ops/ .
RUN cargo build --release

FROM golang:1.25-bookworm AS go-builder
WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .
COPY --from=rust-builder /app/rust_ops/target/release/librust_ops.a rust_ops/target/release/librust_ops.a

RUN CGO_ENABLED=1 go build -o sidecar ./cmd/sidecar

FROM debian:bookworm-slim
WORKDIR /app
COPY --from=go-builder /app/sidecar .

EXPOSE 50051 9090
CMD ["./sidecar"]