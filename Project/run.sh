#!/usr/bin/env bash
# run.sh - deterministic demo: create DB, run demo command, print expected output
set -e
export PYTHONPATH=.
python -c "from Project.app import create_and_insert_demo; print(create_and_insert_demo())"
echo "EXPECTED: inserted: demo-user"