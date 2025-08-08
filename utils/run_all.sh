#!/bin/bash
# Loop through subjects 1 to 128
for i in {1..128}; do
    echo "Processing subject $i"
    python act_processing.py "$i"
done
