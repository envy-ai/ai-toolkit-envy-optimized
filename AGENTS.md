## Instruction inheritance

Before working in this project, read `/home/bart/AGENTS.md` first, then read this project `AGENTS.md`. Apply both sets of instructions; where they conflict, this project file takes precedence. Higher-priority system/developer instructions and permission boundaries still apply.

# Shared instructions

Read and follow `~/AGENTS.md` in addition to this file. If the two files
conflict, the instructions in this project file take precedence.

# Development environment

Use the `ai-toolkit` Conda environment for all project Python commands:

```bash
conda run -n ai-toolkit python ...
```

# Build verification

After changes that affect the UI or server build, run `npm run build` from
`ui/` before finishing, even if the whole server cannot be restarted. An
inability to restart the server is not a reason to skip the build. Report the
build result and whether the server was restarted separately.

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
