# BSM Spin Analysis: Sensitivity and Hypothesis Testing

This repository contains the code and statistical tools used to perform sensitivity projections and hypothesis testing for Beyond the Standard Model (BSM) physics. Specifically, this framework discriminates between different spin hypotheses (e.g., scalar vs. vector-like candidates) using kinematic distributions and precision observables.

## Repository Structure

* `BSM_Spin_Analysis.ipynb`: The main Jupyter Notebook containing the full pipeline — data loading, kinematic distributions, statistical treatment (Asimov datasets, $\chi^2$ evaluations, and $Z$-score calculations), and final results generation.
* `helper.py`: A Python module containing auxiliary functions for statistical calculations, normalization, binning, and plotting utilities used across the analysis.
* `Presentation_slides.pdf`: The official presentation slides detailing the physical motivation, mathematical framework, and key results of the spin discrimination analysis.

---

## Getting Started & Data Setup

Due to file size limits, the simulated Monte Carlo distributions and data files are hosted externally on Google Drive. To run the analysis, you must download the dataset and link it to your environment.

### Step 1: Download the Data
1. Access the dataset via the following Google Drive link:
    **https://drive.google.com/drive/folders/1XXAkBqvWX6iiYFSMeAb-wdzKgTwuUMXi?usp=sharing**
2. Download the entire folder (if downloading from Google Drive via browser, it may download as one or more `.zip` files; make sure to extract and merge them into a single folder).
3. If you are using **Google Colab**, upload the extracted folder to your personal Google Drive so you can mount it directly in the notebook. If you are running **locally**, place the folder anywhere on your local machine.

### Step 2: Configure the Data Paths
Before running the notebook, you must update the file paths to point to where you stored the downloaded data.

1. Open `BSM_Spin_Analysis.ipynb`.
2. Locate the **Data Loading / Configuration** section near the top of the notebook.
3. Alter the `DATA_DIR` (or equivalent path variable) to match your local or Google Drive path:

   **Option A: If running locally on your machine (Linux/macOS/Windows)**
   ```python
   # Example for local execution
   DATA_DIR = "/path/to/your/local/folder/distributions/"
