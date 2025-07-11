#!/bin/bash
#SBATCH --job-name=ede_sim
#SBATCH --output=ede_sim_%j.out
#SBATCH --error=ede_sim_%j.err
#SBATCH --mem=500G
#SBATCH --cpus-per-task=50
#SBATCH --time=30:00:00

module load python/3.10  

# Get suffix argument (e.g., _pysm1)
spectype=$1
theory=$2
binnum="2-15"
bindiff=""
mapset="BK18_planck"
#mapset="BK18"
#theory="all"
#theory="ldiff"
#theory="det_polrot"
#theory="no_ede"
#theory="fixed_dust"

# Define full paths using the suffix
real_data="BK18lf"
dataset="BK18lf"
base_dir="chain_store/iso_val_chains/"
bin_tag="bin$binnum"

theoryname="planck_dust"
file_suffix="${theoryname}/test"

param_path="${base_dir}/${real_data}_${spectype}_${bin_tag}_${file_suffix}"
echo $param_path

# Run your script with max_workers implicitly handled
if true ; then
echo n | python source/full_multicomp.py \
    -s 1 -n 500 -c $theory \
    -p "$param_path" \
    -d "$dataset" \
    -b "$binnum" \
    -m "$mapset" \
    --fede "$fede"\
    -t "$spectype" -o 
fi
if true ; then
echo n | python source/full_multicomp.py \
    -s 1 -n 500 -c $theory \
    -p "$param_path" \
    -d "$dataset" \
    -b "$binnum" \
    -m "$mapset" \
    --fede "$fede"\
    -t "$spectype" -q 
fi

