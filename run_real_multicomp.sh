#!/bin/bash
#SBATCH --job-name=ede_sim
#SBATCH --output=ede_sim_%j.out
#SBATCH --error=ede_sim_%j.err
#SBATCH --mem=100G
#SBATCH --cpus-per-task=11
#SBATCH --time=3:00:00

module load python/3.10  

# Get suffix argument (e.g., _pysm1)
spectype=$1
theory=$2
binnum="2-15"
bindiff=$3
#mapset="BK18_planck"
mapset="BK18"
#theory="all"
#theory="ldiff"
#theory="det_polrot"
#theory="no_ede"
#theory="fixed_dust"
fedename=""
if [ -n "$fede" ]; then
    fedename="_fede$3"
fi

# Define full paths using the suffix
real_data="BK18lf"
dataset="BK18lf_norot_allbins"
base_dir="real_chains/"
bin_tag="bin$binnum"
if [ -n "$bindiff" ]; then
    theory=${theory}${bindiff}
else
    bindiff=0
fi
file_suffix="${theory}${fedename}/real"

param_path="${base_dir}/${realdata}_${spectype}_${bin_tag}_${file_suffix}"
echo $param_path

# Run your script with max_workers implicitly handled
if true ; then
python source/full_multicomp.py \
    -s -1 -n -1 -c $theory \
    -p "$param_path" \
    -d "$dataset" \
    -b "$binnum" \
    -m "$mapset" \
    --fede "$fede"\
    --bin_diff "$bindiff"\
    -t "$spectype" -o 
fi

