#!/data/leuven/372/vsc37234/miniforge3/bin/python
from array import array
import gc
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import detrend, get_window, welch

try:
    import community as community_louvain  # pip install python-louvain
except ImportError:
    community_louvain = None


SEED = 42
plt.rcParams["font.size"] = 14
rng = np.random.default_rng(SEED)
random.seed(SEED)


# -----------------------------
# Scan configuration
# -----------------------------
V = 200
d = 10
N_EDGES = V * d // 2

noise_level = 0.0
ep = 0.0
N_repeat = 1
output_csv = "parameter_scan_ar_2.csv"
a_values = np.linspace(3.58, 3.67, 10)
p_base_values = np.linspace(0.09, 0.14, 51)

tau_s_stop_threshold = -1.3
dfa_stop_threshold = 0.85

rewire_every = 5
max_avalanches = 5000
warmup_drop = 1000

mode = "ar"
network = "random"  # "random", "sw", "regular", "modular"


def _coerce_seed(seed):
    if isinstance(seed, np.random.Generator):
        return int(seed.integers(0, np.iinfo(np.int32).max))
    return seed


def _get_rng(seed):
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def generate_modular_network(V, d, n_modules, p_in=0.15, p_out=0.01, seed=None):
    """Generate a simple modular graph."""
    if community_louvain is None:
        raise ImportError("Modular networks require `python-louvain`.")

    local_rng = np.random.default_rng(seed)
    sizes = [V // n_modules] * n_modules
    for i in range(V % n_modules):
        sizes[i] += 1

    modules = {}
    node_index = 0
    for module_id, size in enumerate(sizes):
        for _ in range(size):
            modules[node_index] = module_id
            node_index += 1

    G = nx.Graph()
    G.add_nodes_from(range(V))

    if p_in is None or p_out is None:
        p_out = d / (V * (1 + (n_modules - 1)))
        p_in = p_out * n_modules

    for i in range(V):
        for j in range(i + 1, V):
            p_edge = p_in if modules[i] == modules[j] else p_out
            if local_rng.random() < p_edge:
                G.add_edge(i, j)
    return G


def generate_network(network, V, d, n_modules, p, seed=None):
    """Generate the initial graph and assign random activations."""
    graph_seed = _coerce_seed(seed)
    node_rng = _get_rng(seed)

    if network == "random":
        G = nx.gnm_random_graph(V, N_EDGES, seed=graph_seed)
    elif network == "regular":
        G = nx.random_regular_graph(d, V, seed=graph_seed)
    elif network == "sw":
        G = nx.watts_strogatz_graph(V, d, p, seed=graph_seed)
    elif network == "modular":
        G = generate_modular_network(V, d, n_modules, p_in=0.15, p_out=0.01, seed=graph_seed)
    else:
        raise ValueError("Unknown network: 'random' | 'regular' | 'sw' | 'modular'")

    for n in G.nodes():
        G.nodes[n]["activation"] = node_rng.uniform(0.0, 1.0)
    return G


def update_activation(G, a, noise_level, ep):
    """Advance node activations by one coupled logistic-map step."""
    new_values = {}
    for n in G.nodes():
        x_n = G.nodes[n]["activation"]
        nbrs = list(G.neighbors(n))
        self_update = a * x_n * (1 - x_n)

        if nbrs:
            nbr_updates = [a * G.nodes[m]["activation"] * (1 - G.nodes[m]["activation"]) for m in nbrs]
            new_values[n] = (1 - ep) * self_update + ep * np.mean(nbr_updates)
        else:
            new_values[n] = self_update

    if noise_level == 0:
        for n, value in new_values.items():
            G.nodes[n]["activation"] = value
    else:
        for n, value in new_values.items():
            G.nodes[n]["activation"] = value + noise_level * rng.random()


def rewiring_no(G):
    return


def rewiring_random_degree_preserving(G, trials=1):
    """Random double-edge swap that preserves node degrees exactly."""
    edges = list(G.edges())
    if len(edges) < 2:
        return

    for _ in range(trials):
        (a, c) = random.choice(edges)
        (b, d) = random.choice(edges)
        if len({a, b, c, d}) < 4:
            continue
        if G.has_edge(a, d) or G.has_edge(b, c):
            continue

        G.remove_edge(a, c)
        G.remove_edge(b, d)
        G.add_edge(a, d)
        G.add_edge(b, c)
        edges = list(G.edges())


def rewiring_random(G):
    """Reconnect one random node to a random non-neighbor."""
    a = random.choice(list(G.nodes()))
    candidates = [u for u in G.nodes() if u != a]
    if not candidates:
        return

    b = random.choice(candidates)
    if not G.has_edge(a, b):
        G.add_edge(a, b)
        nbrs = list(G.neighbors(a))
        if len(nbrs) > 1:
            cands = [u for u in nbrs if u != b] or [b]
            c = random.choice(cands)
            if c != b and G.has_edge(a, c):
                G.remove_edge(a, c)


def rewiring_ar(G):
    """Adaptive rewiring toward the most similar activation state."""
    a = random.choice(list(G.nodes()))
    act_a = G.nodes[a]["activation"]
    b = min(
        (u for u in G.nodes() if u != a),
        key=lambda u: abs(G.nodes[u]["activation"] - act_a),
    )

    if not G.has_edge(a, b):
        G.add_edge(a, b)
        nbrs = list(G.neighbors(a))
        c = max(nbrs, key=lambda u: abs(G.nodes[u]["activation"] - act_a))
        G.remove_edge(a, c)


def do_rewiring(G, mode):
    if mode == "no":
        rewiring_no(G)
    elif mode == "random":
        rewiring_random(G)
    elif mode == "random-dp":
        rewiring_random_degree_preserving(G)
    elif mode == "ar":
        rewiring_ar(G)
    else:
        raise ValueError("Unknown mode")


def one_avalanche(G, p_base):
    """Simulate one avalanche and return size, duration, and activity series."""
    seed_node = int(rng.integers(0, G.number_of_nodes()))
    active = {seed_node}
    size = 0
    duration = 0
    series = []

    while active:
        n_active = len(active)
        series.append(n_active)
        size += n_active
        duration += 1

        next_active = set()
        for u in active:
            x_u = G.nodes[u]["activation"]
            for v in G.neighbors(u):
                x_v = G.nodes[v]["activation"]
                if rng.random() < p_base * (1 - abs(x_u - x_v)):
                    next_active.add(v)
        active = next_active

    return size, duration, series


def dfa_alpha(x, min_window=4, max_frac=0.25, n_scales=12):
    """Simple DFA estimate of the long-range temporal correlation exponent."""
    x = np.asarray(x, dtype=float)
    if len(x) < 16:
        return np.nan

    x = x - np.mean(x)
    y = np.cumsum(x)
    N = len(y)
    s_min = min_window
    s_max = int(N * max_frac)
    if s_max <= s_min + 1:
        return np.nan

    scales = np.unique(np.logspace(np.log10(s_min), np.log10(s_max), n_scales).astype(int))
    Fs, Ss = [], []
    for s in scales:
        nseg = N // s
        if nseg < 2:
            continue

        rms = []
        t = np.arange(s)
        A = np.vstack([t, np.ones_like(t)]).T
        for k in range(nseg):
            seg = y[k * s:(k + 1) * s]
            coef_a, coef_b = np.linalg.lstsq(A, seg, rcond=None)[0]
            trend = coef_a * t + coef_b
            rms.append(np.sqrt(np.mean((seg - trend) ** 2)))

        if rms:
            Fs.append(np.mean(rms))
            Ss.append(s)

    if len(Fs) < 2:
        return np.nan

    coeff = np.polyfit(np.log10(Ss), np.log10(Fs), 1)
    return float(coeff[0])


def psd_loglog_slope_robust(x, fs=1.0, band=None, use_diff=False, p_lo=15, p_hi=85, n_logbins=24):
    """Estimate the PSD slope on a log-log scale."""
    x = np.asarray(x, dtype=float)
    N = x.size
    if N < 128:
        return np.nan, {"reason": "too_short"}

    x = x - np.mean(x)
    if use_diff:
        x = np.diff(x)
    x = detrend(x, type="constant")

    target_segs = 12
    nperseg = int(2 ** np.floor(np.log2(max(256, N // target_segs))))
    nperseg = max(128, min(nperseg, 4096))
    noverlap = nperseg // 2
    win = get_window("hann", nperseg)
    f, Pxx = welch(x, fs=fs, window=win, nperseg=nperseg, noverlap=noverlap, detrend=False, scaling="density")

    valid = (f > 0) & np.isfinite(Pxx) & (Pxx > 0)
    if np.sum(valid) < 20:
        return np.nan, {"reason": "insufficient_bins"}

    f = f[valid]
    Pxx = Pxx[valid]

    if band is None:
        xf = np.log10(f)
        lo, hi = np.percentile(xf, [p_lo, p_hi])
        band_mask = (xf >= lo) & (xf <= hi)
        band_used = (10 ** lo, 10 ** hi)
    else:
        flo, fhi = band
        band_mask = (f >= flo) & (f <= fhi)
        band_used = (flo, fhi)

    f_band = f[band_mask]
    P_band = Pxx[band_mask]
    if f_band.size < 12:
        return np.nan, {"reason": "narrow_band"}

    logf = np.log10(f_band)
    edges = np.linspace(logf.min(), logf.max(), n_logbins + 1)
    mids, pow_med = [], []
    for k in range(n_logbins):
        mask = (logf >= edges[k]) & (logf < edges[k + 1])
        if np.any(mask):
            mids.append(10 ** np.median(logf[mask]))
            pow_med.append(np.median(P_band[mask]))

    f_b = np.asarray(mids)
    P_b = np.asarray(pow_med)
    ok = (f_b > 0) & (P_b > 0)
    f_b = f_b[ok]
    P_b = P_b[ok]
    if f_b.size < 8:
        return np.nan, {"reason": "few_logbins"}

    X = np.log10(f_b)
    Y = np.log10(P_b)
    slope, intercept = np.polyfit(X, Y, 1)
    yhat = slope * X + intercept
    ss_res = np.sum((Y - yhat) ** 2)
    ss_tot = np.sum((Y - np.mean(Y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    info = {
        "intercept": float(intercept),
        "r2": float(r2),
        "npts": int(f_b.size),
        "band_used": band_used,
        "nperseg": int(nperseg),
        "noverlap": int(noverlap),
        "robust": "ols",
        "used_diff": bool(use_diff),
    }
    return float(slope), info


def loglog_hist_stats(data, nbins, label):
    """Return an effective central log-log slope and the span in decades."""
    data = np.asarray(data, dtype=float)
    data = data[data > 0]
    if data.size == 0:
        return np.nan, 0.0

    min_val, max_val = data.min(), data.max()
    orders_span = float(np.log10(max_val) - np.log10(min_val))

    bins = np.logspace(np.log10(min_val), np.log10(max_val), nbins)
    hist, edges = np.histogram(data, bins=bins, density=True)
    centers = np.sqrt(edges[:-1] * edges[1:])
    mask = hist > 0

    slope = np.nan
    xs = np.log10(centers[mask])
    ys = np.log10(hist[mask])
    if xs.size > 3:
        lo, hi = np.percentile(xs, [15, 85])
        mid = (xs >= lo) & (xs <= hi)
        if mid.sum() > 1:
            slope, _ = np.polyfit(xs[mid], ys[mid], 1)

    print(f"{label}: spans ~{orders_span:.2f} decades (min={min_val:.3g}, max={max_val:.3g})")
    return float(slope) if np.isfinite(slope) else np.nan, orders_span


def run_simulation(a, p_base, noise_level, ep=0.0, return_spans=False):
    """
    Run one parameter point.

    Memory optimization:
    - keep avalanche size and duration in compact NumPy arrays
    - keep global activity in a compact integer array instead of a Python list
    - avoid storing network-history tables that are not used by the scan output
    """
    G = generate_network(network, V, d, n_modules=3, p=0.1, seed=rng)

    sizes = np.empty(max_avalanches, dtype=np.int32)
    durations = np.empty(max_avalanches, dtype=np.int32)
    activity_all = array("I")

    for i in range(max_avalanches):
        update_activation(G, a, noise_level, ep)
        if (i + 1) % rewire_every == 0:
            do_rewiring(G, mode)

        size, duration, series = one_avalanche(G, p_base=p_base)
        sizes[i] = size
        durations[i] = duration
        activity_all.extend(series)

    print(f"Mode={mode}, p_base={p_base:.4f}, a={a:.4f}")

    start = min(warmup_drop, max_avalanches - 1)
    trimmed_sizes = sizes[start:]
    trimmed_durations = durations[start:]
    activity_arr = np.fromiter(activity_all, dtype=np.float64)

    dfa_val = dfa_alpha(activity_arr)
    slope, _ = psd_loglog_slope_robust(activity_arr, fs=1.0, band=(1 / 2000, 0.05), use_diff=False)
    tau_s, span_size = loglog_hist_stats(trimmed_sizes, nbins=30, label=f"[{mode}] Avalanche size distribution")
    tau_d, span_dur = loglog_hist_stats(trimmed_durations, nbins=30, label=f"[{mode}] Avalanche duration distribution")

    del G, sizes, durations, activity_all, activity_arr, trimmed_sizes, trimmed_durations
    gc.collect()

    if return_spans:
        return tau_s, tau_d, span_size, span_dur, dfa_val, slope
    return tau_s, tau_d, dfa_val, slope


shape = (len(a_values), len(p_base_values))
tau_s_matrix = np.full(shape, np.nan)
tau_d_matrix = np.full(shape, np.nan)
dfa_matrix = np.full(shape, np.nan)
slope_matrix = np.full(shape, np.nan)

output_columns = ["a", "p_base", "tau_s", "tau_d", "dfa", "slope"]
pd.DataFrame(columns=output_columns).to_csv(output_csv, index=False)


for i, a in enumerate(a_values):
    for j, p_base in enumerate(p_base_values):
        tau_s_list = []
        tau_d_list = []
        dfa_list = []
        slope_list = []

        print(f"Running [{i},{j}] a={a:.3f}, p={p_base:.3f} ...")

        for repeat_idx in range(N_repeat):
            try:
                tau_s, tau_d, dfa_val, slope = run_simulation(a, p_base, noise_level, ep=ep, return_spans=False)
            except Exception as exc:
                print(f"Error at a={a:.3f}, p={p_base:.3f}, repeat={repeat_idx}: {exc}")
                continue

            tau_s_list.append(tau_s)
            tau_d_list.append(tau_d)
            dfa_list.append(dfa_val)
            slope_list.append(slope)

        tau_s_matrix[i, j] = np.nanmean(tau_s_list) if tau_s_list else np.nan
        tau_d_matrix[i, j] = np.nanmean(tau_d_list) if tau_d_list else np.nan
        dfa_matrix[i, j] = np.nanmean(dfa_list) if dfa_list else np.nan
        slope_matrix[i, j] = np.nanmean(slope_list) if slope_list else np.nan

        pd.DataFrame(
            [{
                "a": a,
                "p_base": p_base,
                "tau_s": tau_s_matrix[i, j],
                "tau_d": tau_d_matrix[i, j],
                "dfa": dfa_matrix[i, j],
                "slope": slope_matrix[i, j],
            }]
        ).to_csv(output_csv, mode="a", header=False, index=False)
        print(f"Saved [{i},{j}] to {output_csv}")

        if (
            np.isfinite(tau_s_matrix[i, j])
            and np.isfinite(dfa_matrix[i, j])
            and tau_s_matrix[i, j] > tau_s_stop_threshold
            and dfa_matrix[i, j] < dfa_stop_threshold
        ):
            print(
                f"Stopping p_base loop for a={a:.3f}: "
                f"tau_s={tau_s_matrix[i, j]:.3f} > {tau_s_stop_threshold} and "
                f"dfa={dfa_matrix[i, j]:.3f} < {dfa_stop_threshold}"
            )
            break


print(f"Finished scan. Results were incrementally saved to {output_csv}")


def plot_heatmap(matrix, title, label, filename):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix,
        xticklabels=np.round(p_base_values, 3),
        yticklabels=np.round(a_values, 3),
        cmap="viridis",
        cbar_kws={"label": label},
    )
    plt.title(title)
    plt.xlabel("p_base")
    plt.ylabel("a")
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


plot_heatmap(tau_d_matrix, "tau_d", "tau_d", "heatmap_tau_d.png")
plot_heatmap(tau_s_matrix, "tau_s", "tau_s", "heatmap_tau_s.png")
plot_heatmap(dfa_matrix, "DFA α", "DFA", "heatmap_dfa.png")
plot_heatmap(slope_matrix, "PSD slope", "PSD slope", "heatmap_psd.png")
