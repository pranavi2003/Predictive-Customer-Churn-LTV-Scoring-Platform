#!/bin/bash
# EMR bootstrap action: installs Python deps on every node.
set -euo pipefail
sudo python3 -m pip install --quiet pyyaml pyarrow pandas
