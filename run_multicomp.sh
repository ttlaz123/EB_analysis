#!/bin/bash
#SBATCH --job-name=ede_sim
#SBATCH --output=ede_sim_%j.out
#SBATCH --error=ede_sim_%j.err
#SBATCH --mem=100G
#SBATCH --cpus-per-task=10
#SBATCH --time=12:00:00

module load python/3.10  

# Get suffix argument (e.g., _pysm1)
spectype=$2
injected="_sig$1"
# Define full paths using the suffix
param_path="/n/holylfs04/LABS/kovac_lab/Users/liuto/ede_chains/zeroeb_bin15_gdust_${spectype}${injected}/sim"
dataset="BK18lf"

# Run your script with max_workers implicitly handled
echo n | python source/full_multicomp.py \
    -s 1 -n 500 -c all \
    -p "$param_path" \
    -d "$dataset" \
    -i $1 \
    -t "$spectype" -o 

python source/full_multicomp.py \
    -s 1 -n 500 -c all \
    -p "$param_path" \
    -d "$dataset" \
    -i $1 \
    -t "$spectype" -q


