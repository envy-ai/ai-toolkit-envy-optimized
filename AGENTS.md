# Development environment

Use the `ai-toolkit` Conda environment for all project Python commands:

```bash
conda run -n ai-toolkit python ...
```

# Network services

This machine is behind a firewall and is accessed remotely over the local
network. Unless the user states otherwise, bind development servers and web
apps to `0.0.0.0` rather than `127.0.0.1` or `localhost`.

# Merging upstream changes

Preserve the fork's memory optimizations when merging upstream changes. If an
upstream change cannot be integrated without losing one, stop and ask the user
which behavior to keep before completing the merge.
