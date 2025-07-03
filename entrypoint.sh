#!/bin/bash
set -e

# Ensure cache directories exist and have correct permissions
sudo mkdir -p /home/transcribe/.cache/matplotlib
sudo chown -R transcribe:transcribe /home/transcribe/.cache

# Execute the original command
exec "$@"
