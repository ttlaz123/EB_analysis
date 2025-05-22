#!/bin/bash
#SBATCH --job-name=ede_sim
#SBATCH --output=ede_sim_%j.out
#SBATCH --error=ede_sim_%j.err
#SBATCH --mem=100G
#SBATCH --cpus-per-task=10
#SBATCH --time=10:00:00

module load python/3.10  

# Get suffix argument (e.g., _pysm1)
spectype=$1
injectedx=$3
dataset=$2
binnum="9-15"
#dataset="BK18lf_fede01_sigl"
dusttype="det_polrot"
#dusttype="fixeddust"
#theory="all"
theory="det_polrot"
if [ -n "$injectedx" ]; then
    injected="_sig$3"
fi


if [ "$dataset" == "BK18lf" ]; then
  simdata="zeroeb"
elif [ "$dataset" == "BK18lf_sigl" ]; then
  simdata="zeroeb"
  dusttype="sigl"
elif [ "$dataset" == "BK18lf_fede01" ]; then
  simdata="fede01"
elif [ "$dataset" == "BK18lf_fede01_sigl" ]; then
  simdata="fede01"
  dusttype="sigl"
else
  echo "Unknown dataset: $dataset"
  simdata="unknown"
fi

# Define full paths using the suffix
base_dir="/n/holylfs04/LABS/kovac_lab/Users/liuto/ede_chains"
bin_tag="bin$binnum"
file_suffix="${dusttype}_${spectype}${injected}/sim"

param_path="${base_dir}/${simdata}_${bin_tag}_${file_suffix}"


if [ -z "$injectedx" ]; then
  injectedx="none"
fi
# Run your script with max_workers implicitly handled
echo n | python source/full_multicomp.py \
    -s 1 -n 500 -c $theory \
    -p "$param_path" \
    -d "$dataset" \
    -i "$injectedx" \
    -b "$binnum" \
    -t "$spectype" -o 

python source/full_multicomp.py \
    -s 1 -n 500 -c $theory \
    -p "$param_path" \
    -d "$dataset" \
    -i "$injectedx" \
    -b "$binnum" \
    -t "$spectype" -q


