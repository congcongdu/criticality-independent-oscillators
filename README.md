# criticality-independent-oscillators

This repository contains simulation and analysis code for the work \textbf{Homeostatic self-tuning of critical neuronal avalanches in random activity networks} by Congcong Du, Alkan kabakcioglu and Cees van Leeuwen.

The two main files are:

- [`Code/main.ipynb`]: interactive notebook for running a single simulation, saving time-series outputs, and generating figures for the random activation model.
- [`Code/self-tuning_model.ipynb`]: interactive notebook for running simulations on self-tuning model.

### The key parameter cell defines:

- network size `V`
- mean degree `d`
- transmission probability parameter `p_base`
- rewiring mode `mode`
- initial network type `network`

### Typical outputs written by the notebook include:

- `network_measures_<mode>_<V>.csv`
- `global_activity_<mode>_<V>.csv`
- `avalanches_<mode>_<V>.csv`
- `avalanches_detailed_<mode>_<V>.csv`
- `activation_series_<mode>_<V>.csv`
- `lrtc_summary_<mode>_<V>.csv`


### Notes
- You can always change 'mode' and 'networks' to get results of different rewiring rules (here we only have 'no' mode, but will be more in future works) and initial topologies.
- PC is not recommanded to run critical window, because it cost large memory and time.

### Citation
If you use this repository in academic work, please cite the corresponding paper once available.
