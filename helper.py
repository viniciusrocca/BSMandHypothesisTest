"""
ttbar_helper.py

A unified, object-oriented toolkit for Beyond the Standard Model (BSM) top-quark 
pair (ttbar) collider analyses. 

This module provides an end-to-end pipeline:
    * TTbarDataLoader: File discovery, cross-section normalization, signal-strength optimization, and memory-efficient DataFrame assembly.
    * HistogramBuilder: 1D event binning, luminosity scaling, and signal-plus-background hypothesis construction.
    * StatEngine: Poisson pseudo-experiment (toy) generation, vectorized Pearson chi^2 calculations, and formal hypothesis testing.
    * ColliderPlotter: Professional High-Energy Physics (HEP) visualization for yield ratios and goodness-of-fit distributions.
"""

import glob
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import chi2, norm


# ==============================================================================
# DATA LOADER & ASSEMBLER
# ==============================================================================

class TTbarDataLoader:
    """
    Handles file discovery, baseline normalization, mass scanning, and 
    memory-optimized DataFrame assembly for ttbar BSM analyses.
    
    This class manages loading raw Monte Carlo (MC) event samples from .npz files,
    normalizes background and fake data to a target integrated luminosity, and 
    performs chi-square minimization scans to find the optimal signal mass and 
    coupling parameters before assembling final Pandas DataFrames.
    """
    
    def __init__(self, base_path='./drive/MyDrive/Distributions', lumi=500.0, sys_err=0.00, zp_mass=3200):
        self.base_path = base_path
        self.lumi = lumi
        self.sys_err = sys_err
        self.zp_mass = zp_mass
        
        # Define the invariant mass (m_tt) bin edges and target signal window.
        # The mask restricts optimization to the region where resonant signals are expected to peak.
        self.bins = np.arange(800., 5600., 100.)
        self.mass_mask = (self.bins[:-1] >= 1500) & (self.bins[:-1] <= 5000)
        self.valid_bin_centers = (self.bins[:-1] + np.diff(self.bins) / 2)
        
        self.files = {}
        self.best_fits = {}
        self.chi2_denom_masked = None
        self.n_fake = None
        
    def discover_files(self):
        """Locates and categorizes all required .npz distribution files across model directories."""
        self.files['VLF'] = list(glob.glob(f'{self.base_path}/VLF/qq2ttbar_gs4_ydm2/mass_scan/*.npz')) + \
                            list(glob.glob(f'{self.base_path}/VLF/gg2ttbar_gs4_ydm2/mass_scan/*.npz'))
        self.files['Scalar'] = list(glob.glob(f'{self.base_path}/Scalar/qq2ttbar_gs4_ydm2/mass_scan/*.npz')) + \
                               list(glob.glob(f'{self.base_path}/Scalar/gg2ttbar_gs4_ydm2/mass_scan/*.npz'))
        self.files['Zprime'] = list(glob.glob(f'{self.base_path}/Zprime/mass_scan/*.npz'))
        self.files['Zprime_20pc'] = list(glob.glob(f'{self.base_path}/Zprime/20pc_width/*.npz'))
        self.files['SM'] = list(glob.glob(f'{self.base_path}/SM/pp2ttbar/bias_article*/*.npz'))
        self.files['FakeData'] = list(glob.glob(f'{self.base_path}/Scalar/qq2ttbar_gs4_ydm2/bias_article/mPsiT_1500_mSDM_1400.npz')) + \
                                 list(glob.glob(f'{self.base_path}/Scalar/gg2ttbar_gs4_ydm2/bias_article/mPsiT_1500_mSDM_1400.npz'))
        
        print("\n--- File Discovery Check ---")
        for k, v in self.files.items():
            print(f"{k:<15} files found: {len(v)}")
            
    def _find_best_mu(self, n_sig_template, n_fake_data, denom):
        """
        Optimizes the signal scaling parameter (mu) by minimizing the chi-square 
        discrepancy between the scaled signal template and the observed fake data.
        
        Depending on the model parametrization, 'mu' can represent an overall 
        cross-section multiplier or the fourth power of the dark matter coupling (yDM^4).
        """
        def objective(mu):
            return np.sum(((mu * n_sig_template - n_fake_data)**2) / denom, dtype=np.float64)

        # Enforce non-negative signal strength (mu >= 0) to preserve physical validity
        res = minimize(objective, x0=[8.0], bounds=[(0.0, None)])
        return res.x[0], res.fun

    def build_baselines_and_scan(self, zp_limit_csv_path='Safe_Limits_Zprime.csv'):
        """
        Constructs the SM background and Fake Data baseline spectra, scales them to the target 
        integrated luminosity, and performs parameter scans across mass points to find global minimums.
        """
        # Read reference Z' cross-sections to determine the benchmark fake data target yield
        zp_limit = pd.read_csv(zp_limit_csv_path)
        S_tt_dict = dict(zip(zp_limit['mZp_GeV'], zp_limit['S_tt_pb']))
        target_xsec = S_tt_dict[self.zp_mass] * 0.95
        target_yield = target_xsec * self.lumi * 1000.0  # Convert pb to fb

        # Assemble and scale the Fake Data baseline spectrum
        self.n_fake = np.zeros(len(self.bins) - 1, dtype=np.float64)
        for f in self.files['FakeData']:
            d = np.load(f, allow_pickle=True)
            h_fake, _ = np.histogram(d['mTT'], bins=self.bins, weights=d['weights'])
            self.n_fake += h_fake * self.lumi * 1000.0

        # Normalize the fake data spectrum within the invariant mass window to match the reference target yield
        factor = target_yield / np.sum(self.n_fake[self.mass_mask])
        self.n_fake = self.n_fake * factor
        print(f"Fake Data correctly normalized to yield: {np.sum(self.n_fake[self.mass_mask]):.2f} in target window.")

        # Assemble the Standard Model (SM) background baseline by averaging over available MC samples
        n_sm = np.zeros(len(self.bins) - 1, dtype=np.float64)
        for f in self.files['SM']:
            d = np.load(f, allow_pickle=True)
            h_sm, _ = np.histogram(d['mTT'], bins=self.bins, weights=d['weights'])
            n_sm += h_sm * self.lumi * 1000.0
        n_sm = n_sm / len(self.files['SM']) if len(self.files['SM']) > 0 else n_sm

        # Pre-calculate the variance denominator for chi-square tests, incorporating systematic uncertainties orthogonally
        self.chi2_denom_masked = (self.n_fake + n_sm) + (self.sys_err * n_sm)**2

        # Execute parameter scans across candidate BSM mass hypotheses
        print("\nRunning Mass Scans...")
        chi2_min = {
            'VLF': (None, np.inf, 0.),
            'Scalar': (None, np.inf, 0.),
            'Zprime': (None, np.inf, 0.),
            'Zprime_20pc': (None, np.inf, 0.),
            'FakeData': (1500.0, 0.0, np.sqrt(factor))
        }

        # Scan Vector-Like Fermion (VLF) models across invariant mass points
        for m in np.arange(1300., 1900., 100.):
            vlf_scan = glob.glob(f'{self.base_path}/VLF/*/mass_scan/mPsiT_{m:.0f}_mSDM_{(m-100.):.0f}.npz')
            n_vlf = np.zeros(len(self.bins) - 1, dtype=np.float64)
            for f in vlf_scan:
                d = np.load(f, allow_pickle=True)
                h, _ = np.histogram(d['mTT'], bins=self.bins, weights=d['weights'])
                n_vlf += h * self.lumi * 1000.0
            
            if np.sum(n_vlf) == 0: 
                continue
            best_mu, chi2_val = self._find_best_mu(n_vlf, self.n_fake, self.chi2_denom_masked)
            # Filter out unphysical coupling constants (yDM >= 7.0 violates perturbative unitarity in this model)
            if np.sqrt(best_mu) < 7.0 and chi2_val < chi2_min['VLF'][1]:
                chi2_min['VLF'] = (m, chi2_val, np.sqrt(best_mu))

        # Scan Scalar dark matter mediator models across invariant mass points
        for m in np.arange(1300., 1700., 100.):
            scalar_scan = glob.glob(f'{self.base_path}/Scalar/*/mass_scan/mPsiT_{m:.0f}_mSDM_{(m-100.):.0f}.npz')
            n_scalar = np.zeros(len(self.bins) - 1, dtype=np.float64)
            for f in scalar_scan:
                d = np.load(f, allow_pickle=True)
                h, _ = np.histogram(d['mTT'], bins=self.bins, weights=d['weights'])
                n_scalar += h * self.lumi * 1000.0
            
            if np.sum(n_scalar) == 0: 
                continue
            best_mu, chi2_val = self._find_best_mu(n_scalar, self.n_fake, self.chi2_denom_masked)
            # Apply perturbative coupling threshold limit for scalar models
            if np.sqrt(best_mu) < 10.1 and chi2_val < chi2_min['Scalar'][1]:
                chi2_min['Scalar'] = (m, chi2_val, np.sqrt(best_mu))

        # Scan Z' gauge boson models (evaluating both narrow 1% and broad 20% decay width scenarios)
        for m in np.arange(3000., 3600., 100.):
            # Evaluate narrow width (1%) scenario
            zp_scan = glob.glob(f'{self.base_path}/Zprime/mass_scan/mZp_{m:.0f}.npz')
            n_Zp = np.zeros(len(self.bins) - 1, dtype=np.float64)
            for f in zp_scan:
                d = np.load(f, allow_pickle=True)
                h, _ = np.histogram(d['mTT'], bins=self.bins, weights=d['weights'])
                n_Zp += h * self.lumi * 1000.0
            if np.sum(n_Zp) > 0:
                best_mu, chi2_val = self._find_best_mu(n_Zp, self.n_fake, self.chi2_denom_masked)
                if chi2_val < chi2_min['Zprime'][1]:
                    chi2_min['Zprime'] = (m, chi2_val, np.sqrt(best_mu))
                    
            # Evaluate broad width (20%) scenario
            zp20_scan = glob.glob(f'{self.base_path}/Zprime/20pc_width/mZp_{m:.0f}.npz')
            n_Zp20 = np.zeros(len(self.bins) - 1, dtype=np.float64)
            for f in zp20_scan:
                d = np.load(f, allow_pickle=True)
                h, _ = np.histogram(d['mTT'], bins=self.bins, weights=d['weights'])
                n_Zp20 += h * self.lumi * 1000.0
            if np.sum(n_Zp20) > 0:
                best_mu, chi2_val = self._find_best_mu(n_Zp20, self.n_fake, self.chi2_denom_masked)
                if chi2_val < chi2_min['Zprime_20pc'][1]:
                    chi2_min['Zprime_20pc'] = (m, chi2_val, np.sqrt(best_mu))

        print("\n--- Global Best Fit Results ---")
        for model, fit in chi2_min.items():
            if fit[0] is not None:
                print(f"{model:12}: Mass = {fit[0]:.0f} GeV | yDM = {fit[2]:.6e} | Chi^2 = {fit[1]:.2f}")

        # Store the optimal parameter configurations for downstream DataFrame assembly
        self.best_fits = {
            'VLF':    {'mPsiT': chi2_min['VLF'][0], 'mSDM': chi2_min['VLF'][0] - 100., 'scale_factor': chi2_min['VLF'][2]},
            'Scalar': {'mST': chi2_min['Scalar'][0], 'mChi': chi2_min['Scalar'][0] - 100., 'scale_factor': chi2_min['Scalar'][2]},
            'Zprime': {'mZp': chi2_min['Zprime'][0], 'scale_factor': chi2_min['Zprime'][2]},
            'Zprime_20pc': {'mZp': chi2_min['Zprime_20pc'][0], 'scale_factor': chi2_min['Zprime_20pc'][2]},
            'FakeData': {'scale_factor': chi2_min['FakeData'][2]}
        }

    def assemble_dataframes(self):
        """
        Loads event arrays exclusively for the optimal mass hypotheses identified during scanning,
        applies scaling factors, and constructs memory-optimized Pandas DataFrames.
        """
        if not self.best_fits:
            raise ValueError("Run build_baselines_and_scan() before assembling DataFrames!")
            
        print("\n--- Assembling DataFrames ---")
        bf = self.best_fits
        opt_files = {
            'VLF': glob.glob(f'{self.base_path}/VLF/*/mass_scan/mPsiT_{bf["VLF"]["mPsiT"]:.0f}_mSDM_{bf["VLF"]["mSDM"]:.0f}.npz'),
            'Scalar': glob.glob(f'{self.base_path}/Scalar/*/mass_scan/mPsiT_{bf["Scalar"]["mST"]:.0f}_mSDM_{bf["Scalar"]["mChi"]:.0f}.npz'),
            'Zprime': glob.glob(f'{self.base_path}/Zprime/mass_scan/mZp_{bf["Zprime"]["mZp"]:.0f}.npz'),
            'Zprime_20pc': glob.glob(f'{self.base_path}/Zprime/20pc_width/mZp_{bf["Zprime_20pc"]["mZp"]:.0f}.npz'),
            'SM': self.files['SM'],
            'FakeData': self.files['FakeData']
        }
        
        # Restrict loading to essential kinematics and weights to prevent memory exhaustion
        KEYS_TO_SUM = ['xsec (pb)', 'n_events']
        KEYS_TO_KEEP = ['mTT', 'weights', 'pT']
        raw_data = {k: {'arrays': {}, 'scalars': {}} for k in opt_files.keys()}

        # Read and extract raw data arrays from the optimized file paths
        for model_key, file_list in opt_files.items():
            for f in file_list:
                aux = np.load(f, allow_pickle=True)
                for key in aux.files:
                    val = aux[key]
                    if key in KEYS_TO_SUM:
                        if key not in raw_data[model_key]['scalars']: 
                            raw_data[model_key]['scalars'][key] = 0.0
                        raw_data[model_key]['scalars'][key] += float(val.item())
                    elif val.ndim == 0 or val.size == 1:
                        if not isinstance(val.item(), dict):
                            raw_data[model_key]['scalars'][key] = val.item()
                    else:
                        if key not in KEYS_TO_KEEP: 
                            continue
                        if key not in raw_data[model_key]['arrays']: 
                            raw_data[model_key]['arrays'][key] = []
                        raw_data[model_key]['arrays'][key].append(val[:])
                aux.close()

        # Concatenate extracted arrays across multiple files for each model category
        for model in raw_data:
            for key, list_of_arrays in raw_data[model]['arrays'].items():
                if list_of_arrays:
                    raw_data[model]['arrays'][key] = np.concatenate(list_of_arrays, axis=0)

        # Normalize SM weights by the number of files to account for split sample generation
        if len(opt_files['SM']) > 0 and 'weights' in raw_data['SM']['arrays']:
            raw_data['SM']['arrays']['weights'] = raw_data['SM']['arrays']['weights'].astype(np.float64) / len(opt_files['SM'])

        # Apply optimized cross-section scaling factors to BSM signal weights
        for model_name, data in raw_data.items():
            arrs = data['arrays']
            if not arrs: 
                continue
            if 'weights' in arrs:
                arrs['weights'] = arrs['weights'].astype(np.float64)
            
            if model_name != 'SM':
                factor = np.float64(self.best_fits.get(model_name, {}).get('scale_factor', 1.0))
                factor_sq = factor ** 2
                if factor_sq != 1.0 and 'weights' in arrs:
                    arrs['weights'] = arrs['weights'] * factor_sq
            
            # Standardize column naming conventions for downstream processing
            if 'weights' in arrs: 
                arrs['weight'] = arrs.pop('weights')
            if 'mTT' in arrs:     
                arrs['m_tt'] = arrs.pop('mTT')

        # Assemble BSM DataFrame by stacking arrays across all signal and fake data hypotheses
        final_dict = {}
        labels_list = []
        for model in ['FakeData', 'Scalar', 'VLF', 'Zprime', 'Zprime_20pc']:
            l = len(raw_data[model]['arrays'].get('weight', []))
            if l > 0: 
                labels_list.append(np.full(l, model))
            
        if labels_list: 
            final_dict['label'] = np.concatenate(labels_list)
        
        keys_to_stack = list(raw_data['SM']['arrays'].keys())
        for key in keys_to_stack:
            arrs_to_concat = []
            for model in ['FakeData', 'Scalar', 'VLF', 'Zprime', 'Zprime_20pc']:
                if key in raw_data[model]['arrays'] and len(raw_data[model]['arrays'][key]) > 0:
                    arrs_to_concat.append(raw_data[model]['arrays'][key])
            if arrs_to_concat:
                final_dict[key] = np.concatenate(arrs_to_concat, axis=0)

        df_bsm = pd.DataFrame(final_dict)
        df_sm = pd.DataFrame(raw_data['SM']['arrays'])
        if not df_sm.empty: 
            df_sm['label'] = 'SM'

        # Force garbage collection to free system RAM from large temporary arrays
        del raw_data, final_dict, labels_list
        gc.collect()
        
        print(f"BSM Events: {len(df_bsm)}")
        print(f"SM Events:  {len(df_sm)}")
        return df_bsm, df_sm


# ==============================================================================
# HISTOGRAM BUILDER & SCALER
# ==============================================================================

class HistogramBuilder:
    """
    Decouples event binning and luminosity scaling from plotting routines.
    
    Processes unbinned event DataFrames into 1D distributions, scales event weights 
    to target integrated luminosities, computes combined (SM + BSM) signal hypotheses, 
    and evaluates both statistical and systematic uncertainties.
    """
    
    @staticmethod
    def build_distributions(df_bsm, df_sm, var, lum=500.0, bins=60, rng=None, density=False, sys_err=0.00):
        """
        Bins MC events, applies luminosity scaling, and computes SM + BSM combinations.
        Returns a structured dictionary ready for plotting or Poisson toy sampling.
        """
        label_col = "label" if "label" in df_bsm.columns else "model"
        weight_col_bsm = "weight"
        weight_col_sm = "weight"
        target_bsm_models = ["FakeData", "Scalar", "VLF", "Zprime", "Zprime_20pc"]

        # Drop unphysical infinite values or NaN entries before binning
        df_b_clean = df_bsm[[label_col, var, weight_col_bsm]].replace([np.inf, -np.inf], np.nan).dropna()
        df_s_clean = df_sm[[var, weight_col_sm]].replace([np.inf, -np.inf], np.nan).dropna()

        # Define explicit bin edges based on provided range or dynamic data limits
        if rng is not None:
            bin_edges = np.linspace(rng[0], rng[1], bins + 1)
        else:
            g_min = min(df_b_clean[var].min(), df_s_clean[var].min())
            g_max = max(df_b_clean[var].max(), df_s_clean[var].max())
            bin_edges = np.linspace(g_min, g_max, bins + 1)

        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        saved_distributions = {
            'var': var,
            'lum': lum,
            'sys_err': sys_err,
            'density': density,
            'bin_edges': bin_edges,
            'bin_centers': bin_centers,
            'models': {}
        }

        # Extract, bin, and scale the Standard Model background spectrum
        x_sm = df_s_clean[var].values
        w_sm = df_s_clean[weight_col_sm].values * lum * 1000.0  # Scale weights to target fb^-1

        h_sm, _ = np.histogram(x_sm, bins=bin_edges, weights=w_sm, density=density)
        hErr_sm_poisson = np.sqrt(np.abs(h_sm))
        
        # Exact statistical variance calculation using sum-of-squared weights
        hErr_sm_stat_sq, _ = np.histogram(x_sm, bins=bin_edges, weights=w_sm**2)
        
        # Combine statistical and systematic uncertainties orthogonally in quadrature
        hErr_sm = np.sqrt(hErr_sm_poisson**2 + (sys_err * h_sm)**2)

        saved_distributions['models']['SM'] = {
            'yields': h_sm,
            'err_total': hErr_sm,
            'err_poisson': hErr_sm_poisson,
            'err_stat_only': np.sqrt(hErr_sm_stat_sq),
            'lbl': f"SM (Total: {np.sum(w_sm):.1f})"
        }

        # Extract, bin, and sum BSM models with the SM background to form alternative hypotheses (SM + BSM)
        for lab, sub in df_b_clean.groupby(label_col):
            lab_str = lab.decode('utf-8') if isinstance(lab, bytes) else str(lab)
            if lab_str not in target_bsm_models:
                continue

            x = sub[var].values
            w = sub[weight_col_bsm].values * lum * 1000.0

            h_bsm_only, _ = np.histogram(x, bins=bin_edges, weights=w, density=density)
            err_bsm_only = np.sqrt(np.abs(h_bsm_only))

            # Sum BSM signal with SM background to create the total expected yield for hypothesis testing
            h_total = h_sm + h_bsm_only
            err_total = np.sqrt(hErr_sm**2 + err_bsm_only**2)

            saved_distributions['models'][lab_str] = {
                'yields_total': h_total,       # Total expected rate (lambda) used for Poisson sampling
                'yields_bsm_only': h_bsm_only,
                'err_total': err_total,
                'err_bsm_only': err_bsm_only,
                'lbl': f"{lab_str} + SM (Total: {np.sum(h_total):.1f})" if lab_str != 'FakeData' else f"FakeData ({np.sum(h_bsm_only):.1f})"
            }

        return saved_distributions


# ==============================================================================
# STATISTICAL ENGINE
# ==============================================================================

class StatEngine:
    """
    Provides statistical routines including Poisson pseudo-experiment generation, 
    vectorized chi-square goodness-of-fit evaluations, and formal hypothesis testing.
    """
    
    @staticmethod
    def generate_poisson_copies(saved_dict, model_name, n_copies=1000):
        """
        Generates N Poisson-distributed pseudo-experiments (toys) for a specified model hypothesis.
        
        In HEP, expected rates can sometimes drop slightly below zero due to destructive 
        interference terms at Next-to-Leading Order (NLO). Because the Poisson rate parameter 
        lambda must be non-negative, rates are clipped at zero prior to sampling.
        """
        if model_name not in saved_dict['models']:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(saved_dict['models'].keys())}")
        
        if model_name == 'SM':
            expected_yields = saved_dict['models']['SM']['yields']
        else:
            expected_yields = saved_dict['models'][model_name]['yields_total']
        
        # Protect against unphysical negative rates resulting from NLO event weights
        clean_lambda = np.clip(expected_yields, a_min=0.0, a_max=None)
        return np.random.poisson(lam=clean_lambda, size=(n_copies, len(clean_lambda)))

    @staticmethod
    def compute_chi2_for_models(toys_dict, fake_data_yields, saved_dict, variance_type='model_expect'):
        """
        Evaluates the chi-square goodness-of-fit statistic across all generated pseudo-experiments 
        by comparing them against the baseline Fake Data spectrum.
        
        Supports multiple variance definitions for the chi-square denominator:
            * 'model_expect': Pearson chi-square (variance equals expected model yield).
            * 'fake_data': Neyman chi-square (variance equals observed data yield).
            * 'total_error': Variance incorporates both systematic and MC statistical uncertainties.
        """
        chi2_distributions = {}
        
        for model_name, toys in toys_dict.items():
            if variance_type == 'model_expect':
                denom = saved_dict['models']['SM']['yields'] if model_name == 'SM' else saved_dict['models'][model_name]['yields_total']
            elif variance_type == 'fake_data':
                denom = fake_data_yields
            elif variance_type == 'total_error':
                denom = saved_dict['models']['SM']['err_total']**2 if model_name == 'SM' else saved_dict['models'][model_name]['err_total']**2
            else:
                raise ValueError("Invalid variance_type. Choose: 'model_expect', 'fake_data', or 'total_error'.")
            
            # Prevent division by zero in bins with zero expected events
            clean_denom = np.where(denom <= 0, 1e-10, denom)
            
            # Vectorized calculation of Pearson chi-square across all pseudo-experiments simultaneously
            squared_diff = (toys - fake_data_yields)**2
            chi2_per_toy = np.sum(squared_diff / clean_denom, axis=1)
            chi2_distributions[model_name] = chi2_per_toy
            
        return chi2_distributions

    @staticmethod
    def run_hypothesis_test(chi2_dict, k_bins=60, alpha=0.05):
        """
        Evaluates the statistical compatibility of each model hypothesis against the 
        rejection threshold derived from the chi-square cumulative distribution function.
        
        Calculates p-values and converts them to equivalent Gaussian standard deviation 
        significance levels (Z-scores in sigmas) to determine 95% Confidence Level exclusions.
        """
        # Determine the critical chi-square boundary for the specified alpha level and degrees of freedom
        chi2_crit = chi2.ppf(1.0 - alpha, df=k_bins)
        
        print(f"--- Hypothesis Test Results (d.o.f. = {k_bins}, alpha = {alpha}) ---")
        print(f"Critical Chi^2 Threshold: {chi2_crit:.2f}\n")
        
        results_summary = {}
        for model_name, chi2_vals in chi2_dict.items():
            # Use the median toy chi-square to represent expected experimental sensitivity
            observed_stat = np.median(chi2_vals)
            
            # Calculate p-value using the survival function (1 - CDF) of the chi-square distribution
            p_val = chi2.sf(observed_stat, df=k_bins)
            
            # Convert p-value to equivalent Gaussian significance (sigmas) using the inverse normal CDF
            z_score = norm.ppf(1.0 - p_val) if p_val > 0 else np.inf
            status = "REJECTED (Excluded)" if observed_stat > chi2_crit else "ACCEPTED (Compatible)"
            
            print(f"Model: {model_name:<12} | Median Chi^2: {observed_stat:<6.2f} | p-value: {p_val:<8.4e} | Approx Sig: {z_score:<4.1f}σ -> {status}")
            results_summary[model_name] = {
                'median_chi2': observed_stat, 'p_value': p_val, 'z_score': z_score, 'status': status
            }
        return results_summary


# ==============================================================================
# VISUALIZATION ENGINE
# ==============================================================================

class ColliderPlotter:
    """
    Professional High-Energy Physics (HEP) plotting engine.
    
    Generates superimposed step histograms with ratio sub-panels, handles symmetric 
    logarithmic scaling for wide dynamic yield ranges, and displays goodness-of-fit 
    distributions overlaid with theoretical chi-square probability density functions.
    """
    
    COLOR_MAP = {
        'SM': 'purple', 'VLF': 'green', 'Scalar': 'blue', 
        'Zprime': 'red', 'Zprime_20pc': 'cyan', 'FakeData': 'black'
    }

    @classmethod
    def plot_signal_plus_background(cls, saved_dict):
        """
        Plots SM and (SM + BSM) superimposed 1D distributions on the main panel 
        and displays the bin-by-bin ratio relative to the SM baseline on the lower panel.
        """
        bin_edges = saved_dict['bin_edges']
        bin_centers = saved_dict['bin_centers']
        lum = saved_dict['lum']
        var = saved_dict['var']
        sys_err = saved_dict['sys_err']
        density = saved_dict['density']

        h_sm = saved_dict['models']['SM']['yields']
        hErr_sm = saved_dict['models']['SM']['err_total']
        hErr_sm_poisson = saved_dict['models']['SM']['err_poisson']

        fig, (ax_main, ax_ratio) = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                                                gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})

        for lab_str, data in saved_dict['models'].items():
            c = cls.COLOR_MAP.get(lab_str, 'gray')
            y_vals = data['yields'] if lab_str == 'SM' else data['yields_total']

            # Step histograms represent standard collider visualization practice
            ax_main.hist(bin_centers, bins=bin_edges, weights=np.abs(y_vals),
                         histtype="step", linewidth=2.5, label=data['lbl'], color=c, zorder=3)

            if lab_str == 'SM':
                # Pad arrays by duplicating the last element to align step='post' fill bands with bin edges
                h_pad = np.append(y_vals, y_vals[-1])
                err_pad = np.append(hErr_sm, hErr_sm[-1])
                err_poisson_pad = np.append(hErr_sm_poisson, hErr_sm_poisson[-1])
                
                ax_main.fill_between(bin_edges, 0, err_pad, step='post', color=c, alpha=0.3, zorder=2,
                                     label=rf"SM Unc. (Sys {100*sys_err}%)")
                ax_main.fill_between(bin_edges, 0, err_poisson_pad, step='post', color='gray', alpha=0.3, zorder=1,
                                     label=r"SM Unc. (Poisson)")
                ax_ratio.axhline(1.0, color='k', linestyle='--', alpha=0.8, zorder=1)
            else:
                # Compute ratio (SM + BSM)/SM and propagate errors assuming baseline SM variance dominance
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio = np.where(h_sm > 0, y_vals / h_sm, np.nan)
                    bsm_to_sm = data['yields_bsm_only'] / h_sm
                    ratio_err = np.where(h_sm > 0, (1.0 / np.abs(h_sm)) * np.sqrt(data['err_bsm_only']**2 + (bsm_to_sm**2 * hErr_sm**2)), np.nan)

                ratio_padded = np.append(ratio, ratio[-1])
                ax_ratio.step(bin_edges, ratio_padded, where='post', linewidth=1.5, color=c, alpha=1.0, zorder=3)

        ax_main.set_title(rf"$(\mathrm{{SM}} + \mathrm{{BSM}})$ Yield Comparison | $\mathcal{{L}} = {lum}$ fb$^{{-1}}$", fontsize=12)
        ax_main.set_ylabel("Density" if density else "Expected Events / Bin", fontsize=12)
        ax_main.legend(loc="best", framealpha=1.0, fontsize=9)
        
        # Symmetric log scale allows viewing orders-of-magnitude variations while safely handling zero or negative NLO fluctuations
        ax_main.set_yscale('symlog', linthresh=1e-1)
        ax_main.grid(True, linestyle='--', alpha=0.5)

        ax_ratio.set_xlabel(r'$m_{t\bar{t}}$ (GeV)' if var == 'm_tt' else var, fontsize=13)
        ax_ratio.set_ylabel(r"$\frac{\mathrm{SM} + \mathrm{BSM}}{\mathrm{SM}}$", fontsize=14)
        ax_ratio.grid(True, linestyle=':', alpha=0.5)

        plt.tight_layout()
        plt.show()

    @classmethod
    def plot_chi2_distributions(cls, chi2_dict, n_bins_dof, bins=50):
        """
        Plots the computed chi-square distributions across all model pseudo-experiments 
        and overlays the continuous theoretical chi-square probability density function (PDF).
        """
        plt.figure(figsize=(10, 6))
        all_vals = np.concatenate(list(chi2_dict.values()))
        
        # Cut off extreme 0.5% upper tail outliers to preserve clean visual resolution across the peak
        x_min, x_max = 0, np.percentile(all_vals, 99.5)
        bin_edges = np.linspace(x_min, x_max, bins + 1)
        x_dense = np.linspace(x_min, x_max, 500)
        
        # Overlay theoretical chi-square curve parameterized by degrees of freedom (number of observable bins)
        theo_pdf = chi2.pdf(x_dense, df=n_bins_dof)
        plt.plot(x_dense, theo_pdf, 'k--', linewidth=2, label=f'Theoretical $\chi^2$ (d.o.f. = {n_bins_dof})', zorder=5)
        
        for model_name, chi2_vals in chi2_dict.items():
            c = cls.COLOR_MAP.get(model_name, 'gray')
            plt.hist(chi2_vals, bins=bin_edges, density=True, histtype='step', 
                     linewidth=2.5, color=c, label=f"{model_name} (Mean $\chi^2$: {np.mean(chi2_vals):.1f})", zorder=3)
            plt.hist(chi2_vals, bins=bin_edges, density=True, histtype='stepfilled', 
                     color=c, alpha=0.15, zorder=2)

        plt.title(r"$\chi^2$ Goodness-of-Fit Distributions vs. FakeData", fontsize=14, pad=10)
        plt.xlabel(r"$\chi^2$ Test Statistic", fontsize=13)
        plt.ylabel("Probability Density", fontsize=13)
        plt.legend(loc="upper right", fontsize=11, framealpha=0.9)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.xlim(x_min, x_max)
        plt.tight_layout()
        plt.show()