# Experiment configuration

The JSON files in this directory provide configuration shapes and staged
defaults for inspecting the reference snapshot. `model_endpoints.example.json`
uses placeholder environment-variable names and contains no credentials.

For local use, copy it to `model_endpoints.local.json` and set the referenced
environment variables separately. The repository's `.gitignore` excludes
local registry files. The historical endpoint-probing, readiness, and formal
launch scripts previously mentioned here are not included in this repository;
the configuration examples alone do not reproduce the paper's experiments.
