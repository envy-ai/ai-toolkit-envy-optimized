# Development environment

Use the `ai-toolkit` Conda environment for all project Python commands:

```bash
conda run -n ai-toolkit python ...
```

# Network services

This machine is behind a firewall and is accessed remotely over the local
network. Unless the user states otherwise, bind development servers and web
apps to `0.0.0.0` rather than `127.0.0.1` or `localhost`.
