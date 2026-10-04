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

# Tests and external code

Tests are diagnostic evidence, not goals. Do not modify external code merely
to make a test pass or achieve perfect numerical agreement. When external
behavior prevents a perfect pass, document the limitation and measured
differences, then continue the requested work. Fixing external code requires
a separate explicit user request; a failing test alone is not justification.
