#!/bin/zsh
set -eu
cd "${0:A:h}"
if [[ -n "${MEDCL_PYTHON:-}" ]]; then
  exec "$MEDCL_PYTHON" run.py
elif [[ -x /opt/miniconda3/bin/python ]]; then
  exec /opt/miniconda3/bin/python run.py
else
  exec python3 run.py
fi
