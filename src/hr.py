#!/usr/bin/env python3 

import os
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
from astroquery.gaia import Gaia
from astropy.table import Table
import logging

# Silence astroquery's logger
logging.getLogger('astroquery').setLevel(logging.ERROR)

# Load Gaia credentials from environment variables
GAIA_USER_NAME = os.getenv('GAIA_USER_NAME')
GAIA_USER_PASSWORD = os.getenv('GAIA_USER_PASSWORD')

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate H-R Diagram for globular clusters from Gaia data')
parser.add_argument('config', help='Path to YAML configuration file with cluster parameters')
parser.add_argument('--no-cache', action='store_true', help='Ignore cached data and force Gaia query')
args = parser.parse_args()

# Load cluster parameters from YAML file
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

# 1. Load Cluster Parameters
name = config.get('name', 'Cluster')
ra = config['ra']
dec = config['dec']
radius = config['radius']
dist_pc = config['dist_pc']
data_file = config['data_file']
query_default = f"""
SELECT source_id, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, pmra, pmdec
FROM gaiadr3.gaia_source
WHERE 1=CONTAINS(
    POINT('ICRS', ra, dec),
    CIRCLE('ICRS', {ra}, {dec}, {radius})
)
AND pmra BETWEEN 4.0 AND 6.5
AND pmdec BETWEEN -4.0 AND -1.5
AND phot_g_mean_mag IS NOT NULL
AND phot_bp_mean_mag IS NOT NULL
AND phot_rp_mean_mag IS NOT NULL
"""
query = config.get('query', query_default)

print(f"Loading cluster: {name}")
# print(f"Query: {query}")

# 2. Load or Query Data
if not args.no_cache and os.path.exists(data_file):
    print(f"Loading cached results from {data_file}...")
    results = Table.read(data_file, format='fits')
    print(f"Loaded {len(results)} stars from cache.")
else:
    if args.no_cache and os.path.exists(data_file):
        print(f"Ignoring cached file {data_file} due to --no-cache flag.")
    print(f"Launching Gaia DR3 query for cluster {name}...")
    job = Gaia.launch_job_async(query)
    results = job.get_results()
    print(f"Retrieved {len(results)} stars.")
    
    # Save results to cache
    os.makedirs('data', exist_ok=True)
    results.write(data_file, format='fits', overwrite=True)
    print(f"Saved results to {data_file}")

# 3. Data Processing
print(f"Data processing: calculating color index and absolute magnitude for {len(results)} stars...")

# Color index (x-axis)
bp_rp = results['phot_bp_mean_mag'] - results['phot_rp_mean_mag']

# Absolute Magnitude (y-axis) using distance modulus
# M = m - 5 * log10(d) + 5
abs_mag = results['phot_g_mean_mag'] - 5 * np.log10(dist_pc) + 5

# 4. Plotting the H-R Diagram (CMD)
plt.figure(figsize=(8, 10), facecolor='black')
ax = plt.gca()
ax.set_facecolor('black')

# Using hexbin for density visualization in the crowded core
hb = ax.hexbin(bp_rp, abs_mag, gridsize=150, cmap='magma', bins='log', mincnt=1)

# Formatting
ax.invert_yaxis()  # Brighter stars (lower magnitude) at the top
ax.set_xlabel(r'Color Index $(G_{BP} - G_{RP})$', color='white', fontsize=12)
ax.set_ylabel(r'Absolute Magnitude $(M_G)$', color='white', fontsize=12)
ax.set_title(f'H-R Diagram: {name}', color='white', fontsize=15)
ax.tick_params(colors='white')
ax.set_xlim(-0.25, 2)

# Labeling key regions for your astrophysics background
ax.annotate('Red Giant Branch', xy=(2.0, -2), color='cyan', fontsize=10, fontweight='bold')
ax.annotate('Main Sequence', xy=(0.8, 5), color='cyan', fontsize=10, fontweight='bold')
ax.annotate('Turn-off Point', xy=(0.5, 3.5), color='yellow', fontsize=10)

plt.grid(alpha=0.1, color='gray')
plt.savefig(f'hr_diagram_{name.lower().replace(" ", "_")}.png', dpi=600, bbox_inches='tight', facecolor='black', edgecolor='none')