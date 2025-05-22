#!/bin/bash
# Usage:
# ./run_bindiff.sh zeroeb ""        # no extra suffix after eb
# ./run_bindiff.sh zeroeb _sigpos    # with sigpos after eb
# ./run_bindiff.sh zeroeb _signeg    # with signeg after eb
# ./run_bindiff.sh fede01 ""        # etc.

BASE_ROOT="/n/holylfs04/LABS/kovac_lab/Users/liuto/ede_chains"

# First argument: main suffix (e.g. zeroeb, fede01)
MAIN_SUFFIX=$1
# Second argument: extra suffix after 'eb' (can be empty)
EXTRA_SUFFIX=$2

if [ -z "$MAIN_SUFFIX" ]; then
  echo "Usage: $0 <main_suffix> [extra_suffix_after_eb]"
  exit 1
fi

# Construct folder paths with optional extra suffix after 'eb'
FOLDER1="${BASE_ROOT}/${MAIN_SUFFIX}_bin2-8_det_polrot_eb${EXTRA_SUFFIX}"
FOLDER2="${BASE_ROOT}/${MAIN_SUFFIX}_bin9-15_det_polrot_eb${EXTRA_SUFFIX}"

echo "Running bindiff_calc.py with folders:"
echo "  $FOLDER1"
echo "  $FOLDER2"

python source/bindiff_calc.py "$FOLDER1" "$FOLDER2"

