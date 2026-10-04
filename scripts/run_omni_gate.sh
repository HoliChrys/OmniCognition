#!/bin/bash
# Detached omni gate launcher (survives agent task timeouts).
# The process-compose daemon (managed by the repo's ensure_level_pc) will own
# this process natively after the next natural supervisor restart (autostart);
# until then this keeps the gated server up without touching the daemon.
exec process-compose run omni --no-server \
  -f /home/ubuntu/data_tachikoma/pc/global/process-compose.yml
