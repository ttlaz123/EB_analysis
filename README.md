# EB Analysis Toolkit

A Python-based framework for analyzing EDE constraints.

## Features

- **Simulate** EB spectra with and without injected EDE signals  
- **Jointly fit** EB, EE, and BB spectra including dust and miscalibration models  
- **Perform MCMC** sampling of rotation and dust parameters  
- **Visualize** parameter recovery across many realizations (corner plots, χ² histograms)  
- **Publish** interactive HTML summaries (in each dated folder)

## Installation

1. Clone the repository:
git clone https://github.com/ttlaz123/EB_analysis.git
cd EB_analysis

2. Install requirements:

## Usage

Basic command:
python source/full_multicomp --help

Key scripts:
- source/eb_calculations.py: Core calculations
- source/eb_plot_data.py: Data visualization

## Directory Structure

input_data/       # Raw datasets
chains/           # MCMC chain outputs
source/           # Python modules
202*_*/           # Dated html updates
run_multicomp.sh  # Batch processing script

## Documentation

See comments in:
- full_multicomp.py (basic usage)
- bicep_data_consts.py (experiment parameters)


*Note: Some components are under active development as noted in commit history*