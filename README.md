# criticality-independent-oscillators

This repository contains simulation and analysis code for the work \textbf{Critical-like neuronal avalanches in a network of independent oscillators} by Congcong Du and Cees van Leeuwen.

The two main files are:

- [`Code/main.ipynb`]: interactive notebook for running a single simulation, saving time-series outputs, and generating figures.
- [`Code/scan_ar.py`]: batch parameter-scan script for the adaptive rewiring (`ar`) case, you can use other modes as well.

## 1. main.ipynb
### The key parameter cell defines:

- network size `V`
- mean degree `d`
- logistic-map parameter `a`
- coupling strength `ep`
- avalanche probability scale `p_base`
- rewiring mode `mode`
- initial network type `network`

### Typical outputs written by the notebook include:

- `network_measures_<mode>_<V>.csv`
- `global_activity_<mode>_<V>.csv`
- `avalanches_<mode>_<V>.csv`
- `avalanches_detailed_<mode>_<V>.csv`
- `activation_series_<mode>_<V>.csv`
- `lrtc_summary_<mode>_<V>.csv`

## 2. scan_ar.py

```bash
python Code/scan_ar.py
```

The script scans over:

- `a_values`
- `p_base_values`

and appends results to:

```text
parameter_scan_ar.csv
```

It also generates heatmaps for:

- avalanche size exponent
- avalanche duration exponent
- DFA exponent
- PSD slope

## 3. Notes
- You can always change 'mode' and 'networks' to get results of different rewiring rules and initial topologies.
- PC is not recommanded to run scan_ar.py, because it cost large memory and time.

## 4. Citation

If you use this repository in academic work, please cite the corresponding paper once available.
