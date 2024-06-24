FROM oven/bun:latest

COPY . /app

WORKDIR /app

RUN bun install