#!/bin/bash

for i in $(seq -w 1 10); do
    printf -v padded_i "%03d" "$i"
    python multi_spectra_analysis.py -o -n "$i" -p "chains/sim$i/mpl00c" &
done
