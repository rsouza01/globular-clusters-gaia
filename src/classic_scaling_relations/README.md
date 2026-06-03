# 🧭 Project Title

**Reproducing Classical Scaling Relations in Milky Way Globular Clusters**

***

# 🎯 Problem Statement

Globular clusters (GCs) in the Milky Way exhibit well-known correlations between their **structural properties**, **chemical composition**, and **spatial distribution**. These relations encode information about:

* The formation history of the Galaxy
* The distinction between **in-situ** and **accreted** populations
* The connection between cluster evolution and environment

***

## ✅ Your Objective

Using a standard globular cluster catalog (e.g., Harris catalog), you will:

1. **Reconstruct key empirical relations** among globular cluster observables
2. **Identify patterns and correlations** in these relations
3. **Interpret these patterns in a Galactic context**

***

## 🔍 Scientific Questions

You should aim to address questions such as:

* Do metal-poor clusters preferentially reside farther from the Galactic center?
* Are more luminous clusters found in specific regions of the Galaxy?
* Do structural properties (e.g., size) depend on environment or metallicity?

***

# 📦 Dataset

## Required

* Harris Globular Cluster Catalog (2010 edition or updated version)

## Optional (for later extensions)

* Gaia DR3 (proper motions, distances)
* VizieR cross-matching

***

# 🧪 Phase 1 — Data Preparation

### Step 1: Acquire the dataset

* Obtain the Harris catalog in machine-readable format

### Step 2: Understand the variables

Identify and isolate relevant quantities such as:

* Right Ascension (RA), Declination (Dec)
* Distance from Sun
* Metallicity (\[Fe/H])
* Absolute magnitude (Mv)
* Core radius, half-light radius
* Galactic coordinates (if available)

### Step 3: Clean the dataset

* Remove entries with missing or invalid values
* Ensure units are consistent
* Convert columns into usable numeric formats

***

# 🌌 Phase 2 — Coordinate Transformation

### Step 4: Compute Galactocentric distance

* Convert observed quantities into:
  * Distance from Galactic center (R\_gc)

### Step 5: Validate geometry

* Check expected ranges:
  * Inner clusters (\~1–5 kpc)
  * Outer halo (>20 kpc)

***

# 📊 Phase 3 — Reproduce Classical Relations

You should produce the following plots and inspect them carefully.

***

## 📈 Relation 1: Metallicity vs Galactocentric Distance

### Step 6:

* Plot:
  * x-axis: R\_gc
  * y-axis: \[Fe/H]

### Step 7:

* Look for:
  * Radial gradients
  * Possible separation into populations

***

## 🌟 Relation 2: Luminosity vs Metallicity

### Step 8:

* Convert absolute magnitude (Mv) into luminosity (optional but preferred)

### Step 9:

* Plot:
  * Luminosity (or Mv) vs \[Fe/H]

### Step 10:

* Identify:
  * Whether brighter clusters correlate with metallicity

***

## 📏 Relation 3: Size vs Distance

### Step 11:

* Plot:
  * Half-light radius vs R\_gc

### Step 12:

* Analyze:
  * Do outer clusters appear larger (tidal effects)?

***

## 🌐 Relation 4: Spatial Distribution

### Step 13:

* Construct:
  * 2D projection of cluster positions
  * Optional: 3D visualization

### Step 14:

* Identify:
  * Clustering or anisotropies
  * Halo vs disk population structure

***

# 🧠 Phase 4 — Pattern Identification

### Step 15: Identify populations

* Visually inspect plots for:
  * Distinct groups (e.g., metal-rich vs metal-poor)

### Step 16 (optional but encouraged):

* Apply clustering methods:
  * K-means or Gaussian Mixture Models
* Classify clusters into subgroups

***

# 🧾 Phase 5 — Interpretation

### Step 17: Interpret correlations

You should explicitly address:

* Is there a metallicity gradient with radius?
* Are there distinct populations of clusters?
* Do structural properties correlate with environment?

### Step 18: Connect to astrophysics

Relate findings to concepts such as:

* Galactic halo formation
* Accretion events
* Tidal stripping and dynamical evolution

***

# 📑 Deliverables

At the end, you should have:

### ✅ Plots:

* \[Fe/H] vs R\_gc
* Luminosity vs \[Fe/H]
* Size vs R\_gc
* Spatial distribution map

### ✅ Analysis:

* Written interpretation of each relation
* Identification of at least **two distinct globular cluster populations**

***

# 🚀 Optional Extensions (if you feel like pushing)

* Add **error bars and uncertainties**
* Include **Gaia kinematics**
* Compare with **external galaxies**
* Reproduce results from a published paper

***

# 💡 Final mindset

Treat this as:

> “Reverse-engineering how astronomers first discovered structure in the Milky Way halo.”

***

When you're ready, I can:

* Help you get the Harris catalog in a clean format
* Provide skeleton code (minimal hints)
* Or guide you through interpreting your first plots (this is where the real physics begins)
