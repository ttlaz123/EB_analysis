#!/bin/bash

dir_root="/n/holylfs04/LABS/kovac_lab/Users/liuto/ede_chains"
script="python source/auxiliary_scripts/bindiff_calc.py"

# Loop over all bin2-8 and bin9-15 folders
#for dir1 in "$dir_root"/fede01_bin15_gdust_*/; do
#  for dir2 in "$dir_root"/fede01_bin15_gdust_*/; do
#    echo "Running: $script $dir1 $dir2"
#    $script "$dir1" "$dir2"
#  done
#done

for ede in fede01; do
    for inj in _sigbal _signeg _sigpos ""; do
        for suffix1 in all eb nob; do
            for suffix2 in all eb nob; do
                dir1="$dir_root/${ede}_bin15_gdust_${suffix1}${inj}"
                dir2="$dir_root/${ede}_bin15_gdust_${suffix2}${inj}"
                echo "$script $dir1 $dir2"
                $script "$dir1" "$dir2"
            done
        done
    done
done


