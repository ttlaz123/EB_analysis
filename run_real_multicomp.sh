#!/bin/bash
#SBATCH --job-name=ede_sim
#SBATCH --output=ede_sim_%j.out
#SBATCH --error=ede_sim_%j.err
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00

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
fedename=""
fede=$3
if [ -n "$fede" ]; then
    fedename="_fede$3"
fi

# Define full paths using the suffix
real_data="BK18lf"
dataset="BK18lf_norot_allbins"
base_dir="real_chains/"
bin_tag="bin$binnum"
if [ -n "$bindiff" ]; then
    theoryname=${theory}${bindiff}
else
    theoryname=${theory}
    bindiff=0
fi
theoryname="fixedEBdust_betacmb_planck_prior15"
file_suffix="${theoryname}${fedename}/real"

param_path="${base_dir}/${real_data}_${spectype}_${bin_tag}_${file_suffix}"
echo $param_path

# Run your script with max_workers implicitly handled
if true ; then
echo y | python source/full_multicomp.py \
    -s -1 -n -1 -c $theory \
    -p "$param_path" \
    -d "$dataset" \
    -b "$binnum" \
    -m "$mapset" \
    --fede "$fede"\
    --bin_diff "$bindiff"\
    -t "$spectype"\
    -o 
fi

