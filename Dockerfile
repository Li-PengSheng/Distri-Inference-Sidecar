FROM rust:1.85-slim AS rust-builder

WORKDIR /app/rust_ops
COPY rust_ops/ .
RUN cargo build --release

FROM golang:1.25-bookworm AS go-builder
WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .
COPY --from=rust-builder /app/rust_ops/target/release/librust_ops.so rust_ops/target/release/librust_ops.so

RUN CGO_ENABLED=1 go build -o sidecar ./cmd/sidecar

FROM debian:bookworm-slim
WORKDIR /app
COPY --from=go-builder /app/sidecar .

COPY --from=rust-builder /app/rust_ops/target/release/librust_ops.so /usr/local/lib/
RUN ldconfig
CMD ["./sidecar"]