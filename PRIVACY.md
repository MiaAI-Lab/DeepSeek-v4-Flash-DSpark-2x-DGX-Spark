# Privacy

This repository contains runtime code, reproducibility metadata, sanitized
benchmark aggregates, and public documentation. It does not include telemetry,
tracking, analytics, model weights, credentials, private prompts, LAN topology,
hostnames, usernames, or raw session logs.

The runtime makes no callback to r0b0tlab. Network activity is limited to the
operator-selected model/image sources and the OpenAI-compatible API exposed by
the operator. Use `python3 scripts/public_safety_scan.py .` before publishing.
