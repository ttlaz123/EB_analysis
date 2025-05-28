#!/bin/bash

dir_root="/n/holylfs04/LABS/kovac_lab/Users/liuto/ede_chains"
script="python source/auxiliary_scripts/bindiff_calc.py"

# Loop over all bin2-8 and bin9-15 folders
for dir1 in "$dir_root"/*bin2-10*/; do
  for dir2 in "$dir_root"/*bin11-15*/; do
    echo "Running: $script $dir1 $dir2"
    $script "$dir1" "$dir2"
  done
done

