# This code is extended from the study of Dr. Archie J. Huang (https://github.com/arjhuang/pise)
# The manuscript has been accepted in CSCE 2026, Quebec, Canada

import os, time, csv, random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.io

# USER OPTIONS
DATA_MAT = 'data/redlight.mat'   # change this per dataset and the folder where the data set is placed

# Explicit dataset geometry 
# In this case: matrix is 21 x 600 where 21 = segments (space), 600 = seconds (time)
ROWS_ARE = "space"   # space, is defined for clarity of the data set 
DX_M     = 100.0     # road step [m]
DT_S     = 1.0       # time step [s]
X0_M     = 0.0
T0_S     = 0.0

# Scenario toggles (set exactly one True, once can also run mltiple scenarios at once, just set them "True")
SCENARIO_RANDOM   = False   # selects random data points
SCENARIO_FIXED    = False   # represents Eurlerian data settings
SCENARIO_COMBINED = False   # for combination of Eulerian and Langrangian data 
SCENARIO_CAVS     = True    # used for probe data e.g., langrangian data

# Labeled-data control (RANDOM/FIXED)
TRAIN_PCT         = 0.15    # percentage of data 
FIXED_SENSORS_X   = [0, 200, 500, 600, 750, 1000, 1200,  1400, 1700, 2000]  # locations over the grid (highway) where the sensors are placed
FIXED_TOL_X       = 50.0    # it is tolerance, any location falling below this value will not be considered

# Combined scenario sampling knob (RSU corridor filter)
COMBINED_CAV_PEN  = 0.15    # penetration of CAVs or probes vehicles
COMBINED_STATIONS = [0, 1000, 2000]
COMBINED_RANGE    = 200.0

# CAV probe scenario knobs 
CAV_PEN           = 0.15        # fraction of trajectories retained as "CAV probes"
CAV_MIN_LABELS    = 6000        # minimum labels to keep training stable
CAV_MAX_PEN       = 0.30        # cap, maximum penetration, can exceed this value as per user needs
CAV_NOISE_STD     = 0.0         # additive Gaussian noise on observed v_u, 0 disables

# CAV seeding time-window (seconds) 
CAV_T_START_MIN_S  = 10.0
CAV_T_START_MAX_S  = 590.0

# seeding policy to avoid instant boundary -> flat line
CAV_X_SEED_MODE = "full"         # "upstream" or "full"
CAV_X_UPSTREAM_FRAC = 1          # upstream fraction to seed if mode="upstream"

# boundary behavior 
CAV_HOLD_AT_BOUNDARY = False     # False => stop trajectory at boundary (more realistic visually)

# discrete advection details
CAV_STEP_MODE = "stochastic_round"  # "round" (old), "floor", "stochastic_round" (recommended)
CAV_MIN_POS_STEP = 0               # keep 0 to respect low speeds; set 1 to force movement when v>0 (less physical)

# Optional CAV-specific priors/initials for (c0, T) (OFF by default) 
USE_CAV_PARAM_PRIORS = False
CAV_C0_INIT        = 18.0
CAV_T_INIT         = 8.0
CAV_C0_PRIOR       = 20.0
CAV_T_PRIOR        = 10.0

# Run flags, both scenarios will run, if needed, set one to "False"
RUN_DL            = True
RUN_PINN          = True

# Baseline steps; scaled by N_u later (capped)
ADAM_STEPS_DL_BASE   = 2400
ADAM_STEPS_PINN_BASE = 2400
MAX_ADAM_STEPS_DL    = 10000
MAX_ADAM_STEPS_PINN  = 10000

LR_INIT           = 1e-3
LR_DECAY_STEPS    = (4000, 8000)
CLIP_NORM         = 5.0

F_BATCH_PINN_BASE = 4096
N_F_COLLOCATION0  = 10000

# Loss weights
W_DATA_BASE = 1.0
W_T0_BASE   = 0.5
W_CONT      = 0.2
W_VEL       = 0.2
W_IC        = 0.1

# Physics params (Greenshields)
V_FREE        = 30.0
RHO_M_INIT    = 0.1
RHO_M_BOUNDS  = (0.08, 0.22)
FIX_RHO_M     = False

C0_INIT       = 15.0
T_INIT        = 15.0
C0_BOUNDS     = (2.0, 60.0)
T_BOUNDS      = (2.0, 60.0)

USE_C0T_PRIOR = True
C0_PRIOR, T_PRIOR = 25.0, 20.0
PRIOR_W       = 1e-4

# congestion-conditional evaluation threshold (used for capturing the KPI of congestion)
CONG_SPEED_THRESH = 10.0

SEED = 1

# ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.disable_eager_execution()
random.seed(SEED); np.random.seed(SEED); tf.compat.v1.set_random_seed(SEED)

# Utilities
def ensure_dirs():
    os.makedirs('results', exist_ok=True)

def nearest_idx(vec_1d, vals_2col):
    d = np.abs(vals_2col - vec_1d[None, :])
    return np.argmin(d, axis=1)

def plot_field_single(Vmat, x_vec, t_vec, title, out_png, vmin=None, vmax=None,
                      overlay_X=None, overlay_label=None,
                      overlay_trajs=None, overlay_trajs_label=None,
                      show_legend=True):
    """
    REQUIRED ORIENTATION:
      x-axis = time (t)
      y-axis = space (x)

    """
    plt.figure(figsize=(7.2, 4.6))

    im = plt.imshow(
        Vmat.T, interpolation='nearest', cmap='RdBu',
        extent=[float(t_vec.min()), float(t_vec.max()), float(x_vec.min()), float(x_vec.max())],
        origin='lower', aspect='auto', vmin=vmin, vmax=vmax
    )
    plt.colorbar(im)

    # overlays (no labels unless you explicitly want a legend)
    if overlay_X is not None and len(overlay_X) > 0:
        plt.scatter(overlay_X[:,1], overlay_X[:,0], s=10, marker='o',
                    facecolors='none', linewidths=0.8,
                    label=(overlay_label if overlay_label else "Selected probes"))

    if overlay_trajs is not None and len(overlay_trajs) > 0:
        for i, tr in enumerate(overlay_trajs):
            if tr is None or len(tr) == 0:
                continue
            plt.plot(tr[:,1], tr[:,0], linewidth=1.2,
                     label=(overlay_trajs_label if (i == 0 and overlay_trajs_label) else None))

    if show_legend:
        handles, labels = plt.gca().get_legend_handles_labels()
        if len(labels) > 0:
            plt.legend(loc="upper right", frameon=True)

    plt.title(title)
    plt.xlabel("t [s]")
    plt.ylabel("x [m]")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()

def save_hist_csv(hist, path_csv):
    keys = list(hist.keys())
    n = len(hist[keys[0]])
    with open(path_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for i in range(n):
            w.writerow([hist[k][i] for k in keys])

def try_density_weights(X):
    try:
        from sklearn.neighbors import NearestNeighbors
        k = min(8, len(X))
        nbrs = NearestNeighbors(n_neighbors=k).fit(X)
        dists, _ = nbrs.kneighbors(X)
        scale = np.maximum(dists[:,1:].mean(axis=1), 1e-6)
        w = 1.0 / scale
        return (w / w.mean()).astype(np.float32)[:, None]
    except Exception:
        return np.ones((X.shape[0],1), np.float32)

def sample_from_pool(pool_X, x_vec, t_vec, V):
    xi = nearest_idx(x_vec.squeeze(), pool_X[:,0:1])
    ti = nearest_idx(t_vec.squeeze(), pool_X[:,1:2])
    vals = V[ti, xi][:, None]
    ok = np.isfinite(vals[:,0])
    return pool_X[ok].astype(np.float32), vals[ok].astype(np.float32)

def masked_rmse(A, B, mask):
    if mask is None or not np.any(mask):
        return float("nan")
    diff = (A[mask] - B[mask]).astype(np.float64)
    return float(np.sqrt(np.mean(diff*diff)))

def masked_mae(A, B, mask):
    if mask is None or not np.any(mask):
        return float("nan")
    diff = np.abs((A[mask] - B[mask]).astype(np.float64))
    return float(np.mean(diff))

def masked_rel_l2(A, B, mask):
    if mask is None or not np.any(mask):
        return float("nan")
    denom = float(np.linalg.norm(A[mask].astype(np.float64)))
    if denom < 1e-12:
        return float("nan")
    num = float(np.linalg.norm((A[mask] - B[mask]).astype(np.float64)))
    return float(num / denom)

def compute_velocity_metrics(V_true, V_pred, v_thresh):
    mask_all = np.isfinite(V_true)
    rmse_all = masked_rmse(V_true, V_pred, mask_all)
    mae_all  = masked_mae(V_true, V_pred, mask_all)

    mask_cong = mask_all & (V_true < float(v_thresh))
    rmse_cong = masked_rmse(V_true, V_pred, mask_cong)
    mae_cong  = masked_mae(V_true, V_pred, mask_cong)
    rel_cong  = masked_rel_l2(V_true, V_pred, mask_cong)

    denom = int(np.sum(mask_all))
    cong_frac = float(np.sum(mask_cong) / denom) if denom > 0 else float("nan")

    return rmse_all, mae_all, rmse_cong, mae_cong, rel_cong, cong_frac


# Simplified dataset loader (explicit axes)
def load_v_and_build_axes(mat_path,
                          v_key_candidates=("v","V","vel","velocity","u","speed"),
                          rows_are="space",   # "space" => V_file is (Nx,Nt), "time" => (Nt,Nx)
                          dx_m=100.0,
                          dt_s=1.0,
                          x0_m=0.0,
                          t0_s=0.0):
    data = scipy.io.loadmat(mat_path)
    keys = [k for k in data.keys() if not k.startswith("__")]
    print("MAT keys:", keys)

    v_key = next((k for k in v_key_candidates if k in data), None)
    if v_key is None:
        raise KeyError(f"No velocity matrix found. Tried {v_key_candidates}. Available: {keys}")

    V_file = np.array(data[v_key], dtype=np.float32)
    if V_file.ndim != 2:
        raise ValueError(f"Velocity matrix must be 2D; got shape {V_file.shape}")

    if rows_are.lower() == "space":
        Nx, Nt = V_file.shape
        V = V_file.T  # -> (Nt, Nx)
    elif rows_are.lower() == "time":
        Nt, Nx = V_file.shape
        V = V_file
    else:
        raise ValueError("rows_are must be 'space' or 'time'")

    x_vec = (x0_m + dx_m * np.arange(Nx, dtype=np.float32)).reshape(-1, 1)
    t_vec = (t0_s + dt_s * np.arange(Nt, dtype=np.float32)).reshape(-1, 1)

    print(f"Loaded '{v_key}' from {mat_path}")
    print(f"V_file shape: {V_file.shape} | Using V shape (Nt, Nx): {V.shape}")
    print(f"dx={dx_m} m, dt={dt_s} s")
    print(f"x: {x_vec[0,0]} .. {x_vec[-1,0]} (Nx={Nx})")
    print(f"t: {t_vec[0,0]} .. {t_vec[-1,0]} (Nt={Nt})")

    return V, x_vec, t_vec, data

# Scenarios
def sample_random(V, x_vec, t_vec, pct):
    X, T = np.meshgrid(x_vec.squeeze(), t_vec.squeeze())
    pool = np.column_stack([X.reshape(-1), T.reshape(-1)])
    xi = nearest_idx(x_vec.squeeze(), pool[:,0:1])
    ti = nearest_idx(t_vec.squeeze(), pool[:,1:2])
    ok = np.isfinite(V[ti, xi])
    pool = pool[ok]
    num_valid = pool.shape[0]
    N = max(200, int(pct * num_valid))
    N = min(N, num_valid)
    idx = np.random.choice(num_valid, size=N, replace=False)
    return sample_from_pool(pool[idx], x_vec, t_vec, V)

def sample_fixed(V, x_vec, t_vec, sensors_x, tol_x, pct):
    X, T = np.meshgrid(x_vec.squeeze(), t_vec.squeeze())
    Xs = []
    for s in sensors_x:
        mask = np.isclose(X, s, atol=tol_x)
        if np.any(mask):
            pts = np.column_stack([X[mask], T[mask]])
            Xs.append(pts)
    pool = np.vstack(Xs) if Xs else np.empty((0,2), dtype=np.float32)
    pool = np.unique(np.round(pool, 6), axis=0)
    num_valid = pool.shape[0]
    N = max(200, int(pct * num_valid))
    N = min(N, num_valid)
    if N == 0:
        return np.empty((0,2), np.float32), np.empty((0,1), np.float32)
    idx = np.random.choice(num_valid, size=N, replace=False)
    return sample_from_pool(pool[idx], x_vec, t_vec, V)

def sample_combined(V, x_vec, t_vec, cav_pen, stations_x, rng_x):
    X, T = np.meshgrid(x_vec.squeeze(), t_vec.squeeze())
    stations = list(stations_x) + [float(x_vec.max())]
    rng = float(rng_x)

    mask = False
    for s in stations[:-1]:
        mask = mask | ((s - rng < X) & (X < s + rng))
    mask = mask | (X > stations[-2] - rng)

    pool = np.column_stack([X[mask], T[mask]])
    pool = np.unique(np.round(pool, 6), axis=0)

    frac = float(np.clip(cav_pen, 0.005, 0.25))
    num_valid = pool.shape[0]
    N = max(400, int(frac * num_valid))
    N = min(N, num_valid)
    if N == 0:
        return np.empty((0,2), np.float32), np.empty((0,1), np.float32)
    idx = np.random.choice(num_valid, size=N, replace=False)
    return sample_from_pool(pool[idx], x_vec, t_vec, V)

# CAVS: discrete advection, with EVEN seeding over the full grid 
def sample_cavs_oldstyle(V, x_vec, t_vec, cav_pen, min_labels=400, max_pen=0.25,
                         n_total_base=120,
                         t_start_min_s=10.0, t_start_max_s=590.0,
                         x_seed_mode="upstream", x_upstream_frac=0.35,
                         hold_at_boundary=False,
                         step_mode="stochastic_round",
                         min_pos_step=0):

    cav_pen_eff = float(np.clip(cav_pen, 0.001, max_pen))
    nt, nx = V.shape

    # robust dt, dx from axes
    dt = float(np.median(np.diff(t_vec.squeeze()))) if t_vec.shape[0] > 1 else 1.0
    dx = float(np.median(np.diff(x_vec.squeeze()))) if x_vec.shape[0] > 1 else 1.0
    dt = max(dt, 1e-6)
    dx = max(dx, 1e-6)

    # clip time window
    t_min = float(t_vec.min())
    t_max = float(t_vec.max())
    ts0 = float(np.clip(t_start_min_s, t_min, t_max))
    ts1 = float(np.clip(t_start_max_s, t_min, t_max))
    if ts1 < ts0:
        ts0, ts1 = ts1, ts0

    i_start_min = int(np.argmin(np.abs(t_vec.squeeze() - ts0)))
    i_start_max = int(np.argmin(np.abs(t_vec.squeeze() - ts1)))
    i_start_min = int(np.clip(i_start_min, 0, nt - 2))
    i_start_max = int(np.clip(i_start_max, i_start_min, nt - 2))

    # estimate required number of trajectories (keep your original min-label logic)
    mean_start = 0.5 * (i_start_min + i_start_max)
    mean_len = max(1.0, nt - mean_start)
    min_traj = int(np.ceil(float(min_labels) / mean_len))
    min_traj = max(1, min_traj)

    n_total = max(int(n_total_base), int(np.ceil(min_traj / cav_pen_eff)))
    n_cav = int(np.ceil(cav_pen_eff * n_total))
    n_cav = max(n_cav, min_traj)

    # cap by maximum unique start pairs available in the time window
    ntwin = int(i_start_max - i_start_min + 1)
    max_unique = ntwin * nx
    n_cav = int(min(n_cav, max_unique))

    rng = np.random.RandomState(SEED)

    # ---- build an even lattice in index space (ix, it0) ----
    def _lattice_dims(target, nx_, ntwin_):
        eps = 1e-6
        ratio = (nx_ + eps) / (ntwin_ + eps)
        n_x = max(1, int(np.ceil(np.sqrt(target * ratio))))
        n_t = max(1, int(np.ceil(float(target) / n_x)))
        return n_x, n_t

    n_x_seed, n_t_seed = _lattice_dims(n_cav, nx, ntwin)

    # expand until unique rounded indices provide enough pairs (or we hit full resolution)
    while True:
        ix_grid = np.unique(np.round(np.linspace(0, nx - 1, n_x_seed)).astype(int))
        it_grid = np.unique(np.round(np.linspace(i_start_min, i_start_max, n_t_seed)).astype(int))
        total_pairs = int(len(ix_grid) * len(it_grid))
        if total_pairs >= n_cav or (len(ix_grid) == nx and len(it_grid) == ntwin):
            break
        # expand the smaller dimension first for uniform coverage
        if len(ix_grid) <= len(it_grid):
            n_x_seed += 1
        else:
            n_t_seed += 1

    # snake-order traversal avoids row-major bias
    pairs = []
    for r, it0 in enumerate(it_grid):
        if r % 2 == 0:
            for ix0 in ix_grid:
                pairs.append((int(ix0), int(it0)))
        else:
            for ix0 in ix_grid[::-1]:
                pairs.append((int(ix0), int(it0)))

    # evenly subsample exactly n_cav start points (deterministic, reproducible)
    if len(pairs) > n_cav:
        pick_idx = np.round(np.linspace(0, len(pairs) - 1, n_cav)).astype(int)
        start_pairs = [pairs[i] for i in pick_idx]
    else:
        start_pairs = pairs

    def step_from_speed(v_here):
        s = max(float(v_here), 0.0) * dt / dx  # expected cell movement
        if step_mode == "floor":
            st = int(np.floor(s))
        elif step_mode == "round":
            st = int(np.round(s))
        else:
            base = int(np.floor(s))
            frac = float(s - base)
            st = base + (1 if rng.rand() < frac else 0)

        if v_here > 0.0 and min_pos_step > 0:
            st = max(st, int(min_pos_step))
        return int(max(st, 0))

    cav_trajs = []
    X_list, v_list = [], []

    for (ix0, it0) in start_pairs:
        ix0 = int(np.clip(ix0, 0, nx - 1))
        it0 = int(np.clip(it0, 0, nt - 1))

        traj_pts = []
        cur_ix = ix0

        for it in range(it0, nt):
            x = float(x_vec[cur_ix, 0])
            t = float(t_vec[it, 0])

            v_here = float(V[it, cur_ix]) if np.isfinite(V[it, cur_ix]) else 0.0

            traj_pts.append([x, t])
            X_list.append([x, t])
            v_list.append([v_here])

            if cur_ix >= nx - 1:
                if hold_at_boundary:
                    cur_ix = nx - 1
                    continue
                else:
                    break

            st = step_from_speed(v_here)
            cur_ix = int(np.clip(cur_ix + st, 0, nx - 1))

        traj = np.array(traj_pts, dtype=np.float32)
        if traj.shape[0] >= 2:
            cav_trajs.append(traj)

    X_u = np.array(X_list, dtype=np.float32)
    v_u = np.array(v_list, dtype=np.float32)
    return X_u, v_u, cav_trajs


# Models
class DLOnly:
    def __init__(self, X_u, v_u, X_t0, v_t0, layers, lb, ub, lr=1e-3):
        self.lb, self.ub = lb.astype(np.float32), ub.astype(np.float32)
        self.x_u = X_u[:,0:1].astype(np.float32)
        self.t_u = X_u[:,1:2].astype(np.float32)
        self.v_u = v_u.astype(np.float32)

        self.x_t0 = X_t0[:,0:1].astype(np.float32)
        self.t_t0 = X_t0[:,1:2].astype(np.float32)
        self.v_t0 = v_t0.astype(np.float32)

        self.layers = layers[:]
        self.ws, self.bs = self._init_NN(self.layers)

        self.x_tf  = tf.compat.v1.placeholder(tf.float32, [None,1])
        self.t_tf  = tf.compat.v1.placeholder(tf.float32, [None,1])
        self.v_tf  = tf.compat.v1.placeholder(tf.float32, [None,1])

        self.x_t0_tf = tf.compat.v1.placeholder(tf.float32, [None,1])
        self.t_t0_tf = tf.compat.v1.placeholder(tf.float32, [None,1])
        self.v_t0_tf = tf.compat.v1.placeholder(tf.float32, [None,1])

        self.v_pred     = self._net_v(self.x_tf, self.t_tf)
        self.v_pred_t0  = self._net_v(self.x_t0_tf, self.t_t0_tf)

        mse_data = tf.reduce_mean(tf.square(self.v_tf  - self.v_pred))
        mse_t0   = tf.reduce_mean(tf.square(self.v_t0_tf - self.v_pred_t0))
        self.W_T0_tf = tf.compat.v1.placeholder(tf.float32, shape=())
        self.loss = mse_data + self.W_T0_tf * mse_t0

        opt = tf.compat.v1.train.AdamOptimizer(lr)
        grads_vars = opt.compute_gradients(self.loss)
        clipped    = [(tf.clip_by_norm(g, CLIP_NORM), v) for g,v in grads_vars if g is not None]
        self.train_op = opt.apply_gradients(clipped)

        cfg = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        self.sess = tf.compat.v1.Session(config=cfg)
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def _init_NN(self, layers):
        ws, bs = [], []
        for l in range(len(layers)-1):
            n_in, n_out = layers[l], layers[l+1]
            std = np.sqrt(2.0/(n_in+n_out))
            W = tf.Variable(tf.random.truncated_normal([n_in,n_out], stddev=std, seed=SEED), tf.float32)
            b = tf.Variable(tf.zeros([1,n_out], tf.float32))
            ws.append(W); bs.append(b)
        return ws, bs

    def _neural(self, X):
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for W,b in zip(self.ws[:-1], self.bs[:-1]):
            H = tf.tanh(tf.matmul(H, W) + b)
        return tf.matmul(H, self.ws[-1]) + self.bs[-1]

    def _net_v(self, x, t): return self._neural(tf.concat([x,t], 1))

    def train(self, steps=4000, batch=2048, print_every=200, W_T0_val=0.5):
        N = self.x_u.shape[0]
        t0 = time.time()
        for s in range(1, steps+1):
            idx = np.random.randint(0, N, size=min(batch, N))
            fd = {
                self.x_tf:self.x_u[idx], self.t_tf:self.t_u[idx], self.v_tf:self.v_u[idx],
                self.x_t0_tf:self.x_t0,   self.t_t0_tf:self.t_t0,   self.v_t0_tf:self.v_t0,
                self.W_T0_tf: float(W_T0_val)
            }
            self.sess.run(self.train_op, fd)
            if s % print_every == 0 or s == 1:
                L = self.sess.run(self.loss, fd)
                print(f"[DL  {s:5d}] loss={L:.3e}")
        return time.time() - t0

    def predict(self, X_star):
        return self.sess.run(self._net_v(self.x_tf, self.t_tf),
                             {self.x_tf:X_star[:,0:1].astype(np.float32),
                              self.t_tf:X_star[:,1:2].astype(np.float32)})

class PINN_VelRho:
    """
    Physics:
      Continuity:  rho_t + (rho v)_x = 0
      Velocity:    v_t + (v - c0) v_x - ( V(rho) - v ) / T = 0
      Greenshields: V(rho) = v_f * (1 - rho / rho_m)
    """
    def __init__(self, X_u, v_u, w_u, X_t0, v_t0, X_f, X_ic, rho_ic, layers, lb, ub,
                 v_f,
                 rho_m_init=0.15, rho_m_bounds=(0.08,0.22), fix_rho_m=False,
                 c0_init=15.0, c0_bounds=(2.0,60.0),
                 T_init=15.0,  T_bounds=(2.0,60.0),
                 use_prior=True, prior_c0=25.0, prior_T=20.0, prior_w=1e-4,
                 w_data=1.0, w_cont=0.5, w_vel=0.5, w_ic=0.1, w_t0=0.5,
                 lr=1e-3):

        self.lb, self.ub = lb.astype(np.float32), ub.astype(np.float32)
        self.v_f = float(v_f)

        self.x_u = X_u[:,0:1].astype(np.float32); self.t_u = X_u[:,1:2].astype(np.float32)
        self.v_u = v_u.astype(np.float32)
        self.w_u = w_u.astype(np.float32)

        self.x_t0 = X_t0[:,0:1].astype(np.float32); self.t_t0 = X_t0[:,1:2].astype(np.float32)
        self.v_t0 = v_t0.astype(np.float32)

        self.x_f = X_f[:,0:1].astype(np.float32); self.t_f = X_f[:,1:2].astype(np.float32)

        self.x_ic= X_ic[:,0:1].astype(np.float32); self.t_ic= X_ic[:,1:2].astype(np.float32)
        self.rho_ic = rho_ic.astype(np.float32)

        if fix_rho_m:
            self.rho_m = tf.constant(rho_m_init, dtype=tf.float32)
            self._has_rhom_var = False
        else:
            self.rho_m_var = tf.Variable(rho_m_init, tf.float32)
            lo, hi = rho_m_bounds
            self.rho_m = tf.clip_by_value(self.rho_m_var, lo, hi)
            self._has_rhom_var = True

        self.c0_var = tf.Variable(c0_init, tf.float32)
        self.T_var  = tf.Variable(T_init,  tf.float32)
        self.c0 = tf.clip_by_value(self.c0_var, *c0_bounds)
        self.T  = tf.clip_by_value(self.T_var,  *T_bounds)

        self.layers_v = layers[:]; self.layers_r = layers[:]
        self.wv, self.bv = self._init_NN(self.layers_v)
        self.wr, self.br = self._init_NN(self.layers_r)

        self.x_u_tf  = tf.compat.v1.placeholder(tf.float32, [None,1])
        self.t_u_tf  = tf.compat.v1.placeholder(tf.float32, [None,1])
        self.v_u_tf  = tf.compat.v1.placeholder(tf.float32, [None,1])
        self.w_u_tf  = tf.compat.v1.placeholder(tf.float32, [None,1])

        self.x_t0_tf = tf.compat.v1.placeholder(tf.float32, [None,1])
        self.t_t0_tf = tf.compat.v1.placeholder(tf.float32, [None,1])
        self.v_t0_tf = tf.compat.v1.placeholder(tf.float32, [None,1])

        self.x_f_tf  = tf.compat.v1.placeholder(tf.float32, [None,1])
        self.t_f_tf  = tf.compat.v1.placeholder(tf.float32, [None,1])

        self.x_ic_tf = tf.compat.v1.placeholder(tf.float32, [None,1])
        self.t_ic_tf = tf.compat.v1.placeholder(tf.float32, [None,1])
        self.rho_ic_tf = tf.compat.v1.placeholder(tf.float32, [None,1])

        self.v_pred_u  = self._net_v(self.x_u_tf,  self.t_u_tf)
        self.v_pred_t0 = self._net_v(self.x_t0_tf, self.t_t0_tf)

        v_fld = self._net_v(self.x_f_tf, self.t_f_tf)
        r_fld = self._net_r(self.x_f_tf, self.t_f_tf)

        v_t = tf.gradients(v_fld, self.t_f_tf)[0]
        v_x = tf.gradients(v_fld, self.x_f_tf)[0]
        r_t = tf.gradients(r_fld, self.t_f_tf)[0]
        rv_x= tf.gradients(r_fld * v_fld, self.x_f_tf)[0]

        Veq = self.v_f * (1.0 - r_fld / self.rho_m)
        f_cont = r_t + rv_x
        f_vel  = v_t + (v_fld - self.c0) * v_x - (Veq - v_fld)/self.T

        r_ic_pred = self._net_r(self.x_ic_tf, self.t_ic_tf)

        data_res = self.v_u_tf - self.v_pred_u
        mse_data = tf.reduce_sum(self.w_u_tf * tf.square(data_res)) / (tf.reduce_sum(self.w_u_tf) + 1e-8)
        mse_t0   = tf.reduce_mean(tf.square(self.v_t0_tf - self.v_pred_t0))
        mse_cont = tf.reduce_mean(tf.square(f_cont))
        mse_vel  = tf.reduce_mean(tf.square(f_vel))
        mse_ic   = tf.reduce_mean(tf.square(r_ic_pred - self.rho_ic_tf))

        self.w_data = tf.Variable(w_data, trainable=False, dtype=tf.float32)
        self.w_t0   = tf.Variable(w_t0,   trainable=False, dtype=tf.float32)
        self.w_cont = tf.Variable(w_cont, trainable=False, dtype=tf.float32)
        self.w_vel  = tf.Variable(w_vel,  trainable=False, dtype=tf.float32)
        self.w_ic   = tf.Variable(w_ic,   trainable=False, dtype=tf.float32)

        self.loss = ( self.w_data*mse_data + self.w_t0*mse_t0
                    + self.w_cont*mse_cont + self.w_vel*mse_vel + self.w_ic*mse_ic )

        if use_prior:
            self.loss += prior_w * ((self.c0 - prior_c0)**2 + (self.T - prior_T)**2)

        self.lr = tf.Variable(lr, trainable=False, dtype=tf.float32)
        opt = tf.compat.v1.train.AdamOptimizer(self.lr)
        grads_vars = opt.compute_gradients(self.loss)
        clipped    = [(tf.clip_by_norm(g, CLIP_NORM), v) for g,v in grads_vars if g is not None]
        self.train_op = opt.apply_gradients(clipped)

        self.fetch_losses = [self.loss, mse_data, mse_t0, mse_cont, mse_vel, mse_ic, self.c0, self.T, self.rho_m]
        extra = [self.c0_var, self.T_var]
        if self._has_rhom_var:
            extra += [self.rho_m_var]
        self.fetch_all = self.fetch_losses + extra

        cfg = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        self.sess = tf.compat.v1.Session(config=cfg)
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def _init_NN(self, layers):
        ws, bs = [], []
        for l in range(len(layers)-1):
            n_in, n_out = layers[l], layers[l+1]
            std = np.sqrt(2.0/(n_in+n_out))
            W = tf.Variable(tf.random.truncated_normal([n_in,n_out], stddev=std, seed=SEED), tf.float32)
            b = tf.Variable(tf.zeros([1,n_out], tf.float32))
            ws.append(W); bs.append(b)
        return ws, bs

    def _scale_in(self, X):
        return 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0

    def _net_v(self, x, t):
        H = self._scale_in(tf.concat([x,t],1))
        for W,b in zip(self.wv[:-1], self.bv[:-1]):
            H = tf.tanh(tf.matmul(H, W) + b)
        return tf.matmul(H, self.wv[-1]) + self.bv[-1]

    def _net_r(self, x, t):
        H = self._scale_in(tf.concat([x,t],1))
        for W,b in zip(self.wr[:-1], self.br[:-1]):
            H = tf.tanh(tf.matmul(H, W) + b)
        raw = tf.matmul(H, self.wr[-1]) + self.br[-1]
        return self.rho_m * tf.sigmoid(raw)

    def feed(self, f_idx=None, d_idx=None):
        if f_idx is None:
            xf, tf_ = self.x_f, self.t_f
        else:
            xf, tf_ = self.x_f[f_idx], self.t_f[f_idx]
        if d_idx is None:
            xu, tu, vu, wu = self.x_u, self.t_u, self.v_u, self.w_u
        else:
            xu, tu, vu, wu = self.x_u[d_idx], self.t_u[d_idx], self.v_u[d_idx], self.w_u[d_idx]
        return {
            self.x_u_tf:xu, self.t_u_tf:tu, self.v_u_tf:vu, self.w_u_tf:wu,
            self.x_t0_tf:self.x_t0, self.t_t0_tf:self.t_t0, self.v_t0_tf:self.v_t0,
            self.x_f_tf:xf, self.t_f_tf:tf_,
            self.x_ic_tf:self.x_ic, self.t_ic_tf:self.t_ic, self.rho_ic_tf:self.rho_ic
        }

    def train(self, steps=10000, f_batch=4096, d_batch=2048, print_every=200, lr_decay_steps=(4000,8000)):
        nF = self.x_f.shape[0]
        nD = self.x_u.shape[0]
        t0 = time.time()

        ramp_step = 3000 if steps >= 3000 else max(1, int(0.30 * steps))
        lr_decay_steps = tuple([s for s in lr_decay_steps if s <= steps])

        hist = {
            "step": [], "loss": [], "data": [], "t0": [], "cont": [], "vel": [], "ic": [],
            "c0": [], "T": [], "rho_m": [],
            "c0_raw": [], "T_raw": [], "rho_m_raw": []
        }

        for s in range(1, steps+1):
            f_idx = np.random.randint(0, nF, size=min(f_batch, nF))
            d_idx = np.random.randint(0, nD, size=min(d_batch, nD))
            self.sess.run(self.train_op, self.feed(f_idx, d_idx))

            if s == ramp_step:
                self.sess.run([self.w_cont.assign(self.w_cont*2.0),
                               self.w_vel.assign(self.w_vel*2.0)])

            if s in lr_decay_steps:
                self.sess.run(self.lr.assign(self.lr * 0.3))

            if s % print_every == 0 or s == 1:
                outs = self.sess.run(self.fetch_all, self.feed(f_idx, d_idx))
                L, Ld, Lt0, Lc, Lv, Lic, c0v, Tv, rhom = outs[:9]
                c0_raw = outs[9]
                T_raw  = outs[10]
                rho_raw = outs[11] if len(outs) > 11 else float("nan")

                hist["step"].append(int(s))
                hist["loss"].append(float(L))
                hist["data"].append(float(Ld))
                hist["t0"].append(float(Lt0))
                hist["cont"].append(float(Lc))
                hist["vel"].append(float(Lv))
                hist["ic"].append(float(Lic))
                hist["c0"].append(float(c0v))
                hist["T"].append(float(Tv))
                hist["rho_m"].append(float(rhom))
                hist["c0_raw"].append(float(c0_raw))
                hist["T_raw"].append(float(T_raw))
                hist["rho_m_raw"].append(float(rho_raw))

                print(f"[PINN{s:5d}] loss={L:.3e} | data={Ld:.3e} t0={Lt0:.3e} cont={Lc:.3e} vel={Lv:.3e} ic={Lic:.3e} "
                      f"| c0={c0v:.2f} T={Tv:.2f} rho_m={rhom:.3f}")

        return (time.time() - t0), hist, ramp_step, lr_decay_steps

    def predict_v(self, X_star):
        return self.sess.run(self._net_v(self.x_u_tf, self.t_u_tf),
                             {self.x_u_tf:X_star[:,0:1].astype(np.float32),
                              self.t_u_tf:X_star[:,1:2].astype(np.float32)})


def plot_pinn_diagnostics(hist, scenario, ramp_step, lr_decay_steps):
    steps = np.array(hist["step"], dtype=float)

    plt.figure(figsize=(8.0, 4.8))
    plt.plot(steps, hist["loss"], label="total")
    plt.plot(steps, hist["data"], label="data")
    plt.plot(steps, hist["t0"],   label="t0")
    plt.plot(steps, hist["cont"], label="cont")
    plt.plot(steps, hist["vel"],  label="vel")
    plt.plot(steps, hist["ic"],   label="ic")
    plt.yscale("log")
    plt.axvline(ramp_step, linestyle=":", label="physics ramp")
    for s0 in lr_decay_steps:
        plt.axvline(s0, linestyle=":")
    plt.xlabel("Training step")
    plt.ylabel("Loss (log scale)")
    plt.title(f"PINN loss components [{scenario}]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/loss_pinn_{scenario}.png", dpi=180)
    plt.close()

    fig, axs = plt.subplots(3, 1, figsize=(8.0, 10.5), squeeze=False)

    axs[0,0].plot(steps, hist["c0"], label="c0 (clipped)")
    axs[0,0].plot(steps, hist["c0_raw"], "--", label="c0_raw")
    axs[0,0].axhline(C0_BOUNDS[0], linestyle=":")
    axs[0,0].axhline(C0_BOUNDS[1], linestyle=":")
    axs[0,0].axvline(ramp_step, linestyle=":")
    for s0 in lr_decay_steps: axs[0,0].axvline(s0, linestyle=":")
    axs[0,0].set_ylabel("c0"); axs[0,0].legend()
    axs[0,0].set_title(f"Calibration parameter trajectories [{scenario}]")

    axs[1,0].plot(steps, hist["T"], label="T (clipped)")
    axs[1,0].plot(steps, hist["T_raw"], "--", label="T_raw")
    axs[1,0].axhline(T_BOUNDS[0], linestyle=":")
    axs[1,0].axhline(T_BOUNDS[1], linestyle=":")
    axs[1,0].axvline(ramp_step, linestyle=":")
    for s0 in lr_decay_steps: axs[1,0].axvline(s0, linestyle=":")
    axs[1,0].set_ylabel("T"); axs[1,0].legend()

    axs[2,0].plot(steps, hist["rho_m"], label="rho_m (clipped)")
    axs[2,0].plot(steps, hist["rho_m_raw"], "--", label="rho_m_raw")
    axs[2,0].axhline(RHO_M_BOUNDS[0], linestyle=":")
    axs[2,0].axhline(RHO_M_BOUNDS[1], linestyle=":")
    axs[2,0].axvline(ramp_step, linestyle=":")
    for s0 in lr_decay_steps: axs[2,0].axvline(s0, linestyle=":")
    axs[2,0].set_xlabel("Training step"); axs[2,0].set_ylabel("rho_m"); axs[2,0].legend()

    plt.tight_layout()
    plt.savefig(f"results/trace_params_pinn_{scenario}.png", dpi=180)
    plt.close()

    plt.figure(figsize=(5.5, 5.5))
    plt.plot(hist["c0"], hist["T"], "-o", markersize=2)
    plt.plot([C0_PRIOR], [T_PRIOR], "x", markersize=10)
    plt.xlabel("c0"); plt.ylabel("T")
    plt.title(f"Parameter path (c0 vs T) [{scenario}]")
    plt.tight_layout()
    plt.savefig(f"results/path_c0_T_pinn_{scenario}.png", dpi=180)
    plt.close()


# Main
if __name__ == "__main__":
    ensure_dirs()

    V, x_vec, t_vec, data = load_v_and_build_axes(
        DATA_MAT,
        rows_are=ROWS_ARE,
        dx_m=DX_M, dt_s=DT_S,
        x0_m=X0_M, t0_s=T0_S
    )

    # Grid & bounds
    X, T = np.meshgrid(x_vec.squeeze(), t_vec.squeeze())
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None])).astype(np.float32)
    lb = X_star.min(0); ub = X_star.max(0)

    nt, nx = V.shape

    # Initial-time anchor (row with most finite entries)
    valid_counts = np.sum(np.isfinite(V), axis=1)
    t0_idx = int(np.argmax(valid_counts))
    X_t0 = np.hstack([x_vec.astype(np.float32), np.full_like(x_vec, t_vec[t0_idx], dtype=np.float32)])
    v_t0 = V[t0_idx, :][:, None].astype(np.float32)

    flags = [SCENARIO_RANDOM, SCENARIO_FIXED, SCENARIO_COMBINED, SCENARIO_CAVS]
    assert sum(flags) == 1, "Set exactly one of SCENARIO_RANDOM/FIXED/COMBINED/CAVS = True"

    cav_trajs = None

    if SCENARIO_RANDOM:
        scenario = f"random_{TRAIN_PCT:.3f}"
        X_u, v_u = sample_random(V, x_vec, t_vec, TRAIN_PCT)

    elif SCENARIO_FIXED:
        scenario = f"fixed_{TRAIN_PCT:.3f}"
        X_u, v_u = sample_fixed(V, x_vec, t_vec, FIXED_SENSORS_X, FIXED_TOL_X, TRAIN_PCT)

    elif SCENARIO_COMBINED:
        scenario = f"combined_{COMBINED_CAV_PEN:.3f}"
        X_u, v_u = sample_combined(V, x_vec, t_vec, COMBINED_CAV_PEN, COMBINED_STATIONS, COMBINED_RANGE)

    else:
        cav_pen_eff = float(np.clip(CAV_PEN, 0.001, CAV_MAX_PEN))
        scenario = f"cavs_{cav_pen_eff:.3f}"

        X_u, v_u, cav_trajs = sample_cavs_oldstyle(
            V, x_vec, t_vec,
            cav_pen_eff,
            min_labels=CAV_MIN_LABELS,
            max_pen=CAV_MAX_PEN,
            t_start_min_s=CAV_T_START_MIN_S,
            t_start_max_s=CAV_T_START_MAX_S,
            x_seed_mode=CAV_X_SEED_MODE,
            x_upstream_frac=CAV_X_UPSTREAM_FRAC,
            hold_at_boundary=CAV_HOLD_AT_BOUNDARY,
            step_mode=CAV_STEP_MODE,
            min_pos_step=CAV_MIN_POS_STEP
        )

        if float(CAV_NOISE_STD) > 0.0 and v_u.size > 0:
            v_u = (v_u + np.random.normal(0.0, float(CAV_NOISE_STD), size=v_u.shape)).astype(np.float32)

    print(f"Labeled points N_u: {X_u.shape[0]}")
    assert X_u.shape[0] >= 300, "Too few labels. Increase TRAIN_PCT/CAV_PEN or adjust CAV knobs."

    w_u = try_density_weights(X_u)

    # Collocation points: uniform + data + gradient-aware
    N_u = X_u.shape[0]
    N_F_COLLOCATION = max(N_F_COLLOCATION0, int(8000 + 2.0*N_u))
    X_f = lb + (ub - lb) * np.random.rand(N_F_COLLOCATION, 2).astype(np.float32)
    X_f = np.vstack([X_f, X_u]).astype(np.float32)

    dVdt = np.nan_to_num(np.abs(np.gradient(V, axis=0)), nan=0.0)
    dVdx = np.nan_to_num(np.abs(np.gradient(V, axis=1)), nan=0.0)
    score = (dVdt + dVdx)
    score /= (score.max() + 1e-8)

    Ti, Xi = np.indices((nt, nx))
    flat = np.column_stack([Xi.ravel(), Ti.ravel()])
    prob = score.ravel()
    prob = prob / (prob.sum() + 1e-8)

    k = int(0.4 * N_F_COLLOCATION)
    sel = np.random.choice(flat.shape[0], size=k, replace=False, p=prob)
    xf = x_vec[flat[sel, 0]].astype(np.float32)
    tf_ = t_vec[flat[sel, 1]].astype(np.float32)
    X_f = np.vstack([X_f, np.hstack([xf, tf_])]).astype(np.float32)

    # IC for rho
    N_ic = 300
    x_ic = np.random.uniform(float(x_vec.min()), float(x_vec.max()), size=(N_ic,1)).astype(np.float32)
    t_ic = np.full_like(x_ic, float(t_vec.min()), dtype=np.float32)
    rho_ic = np.full_like(x_ic, 0.03, dtype=np.float32)
    X_ic = np.hstack([x_ic, t_ic]).astype(np.float32)

    # v_free heuristic
    vmin_data = float(np.nanpercentile(V, 1))
    vmax_data = float(np.nanpercentile(V, 99))
    v_free = max(vmax_data, V_FREE)

    layers = [2] + [40]*4 + [1]

    def steps_for_size(Nu, base_steps=2000, ref=2000, cap=20000):
        return int(min(cap, base_steps * max(1.0, Nu / ref)))

    ADAM_STEPS_DL   = steps_for_size(N_u, base_steps=ADAM_STEPS_DL_BASE,   ref=2000, cap=MAX_ADAM_STEPS_DL)
    ADAM_STEPS_PINN = steps_for_size(N_u, base_steps=ADAM_STEPS_PINN_BASE, ref=2000, cap=MAX_ADAM_STEPS_PINN)

    BATCH_DL        = min(2048, N_u)
    F_BATCH_PINN    = min(F_BATCH_PINN_BASE, X_f.shape[0])
    D_BATCH_PINN    = min(2048, N_u)

    if SCENARIO_COMBINED:
        W_T0 = float(W_T0_BASE)
        W_DATA = float(W_DATA_BASE + 1.0)
    elif SCENARIO_CAVS:
        W_T0 = float(np.clip(W_T0_BASE * 0.8, 0.05, 0.5))
        W_DATA = float(np.clip(W_DATA_BASE + 1.0, 1.0, 3.0))
    else:
        W_T0 = float(np.clip(W_T0_BASE * (0.2 / max(1e-3, TRAIN_PCT)), 0.05, 0.5))
        W_DATA = float(np.clip(W_DATA_BASE + 2.0 * TRAIN_PCT, 1.0, 3.0))

    if SCENARIO_CAVS and USE_CAV_PARAM_PRIORS:
        _C0_INIT_USE  = float(CAV_C0_INIT)
        _T_INIT_USE   = float(CAV_T_INIT)
        _C0_PRIOR_USE = float(CAV_C0_PRIOR)
        _T_PRIOR_USE  = float(CAV_T_PRIOR)
    else:
        _C0_INIT_USE  = float(C0_INIT)
        _T_INIT_USE   = float(T_INIT)
        _C0_PRIOR_USE = float(C0_PRIOR)
        _T_PRIOR_USE  = float(T_PRIOR)

    print(f"Shapes: V={V.shape} | X_u={X_u.shape} | X_f={X_f.shape} | X_ic={X_ic.shape} | X_t0={X_t0.shape}")
    print(f"Steps: DL={ADAM_STEPS_DL} (batch {BATCH_DL}) | PINN={ADAM_STEPS_PINN} (f_batch {F_BATCH_PINN}, d_batch {D_BATCH_PINN})")
    print(f"Weights: W_DATA={W_DATA:.2f} W_CONT={W_CONT:.2f} W_VEL={W_VEL:.2f} W_IC={W_IC:.2f} W_T0={W_T0:.2f}")
    print(f"Congestion eval: v < {CONG_SPEED_THRESH:g}")

    # Overlay policy 
    # CAVS figures: ONLY trajectories, NO points, NO legend
    if SCENARIO_CAVS:
        overlay_pts = None
        overlay_lbl = None
        overlay_trajs = cav_trajs
        overlay_trajs_lbl = None
        show_legend_on_field = False
    else:
        overlay_pts = None
        overlay_lbl = None
        overlay_trajs = None
        overlay_trajs_lbl = None
        show_legend_on_field = True

    # Original field (time on x, space on y)
    plot_field_single(V, x_vec, t_vec, f"Reference Velocity (Generated with METANET)", f"results/original_{scenario}.png",
                      vmin=vmin_data, vmax=vmax_data,
                      overlay_X=overlay_pts, overlay_label=overlay_lbl,
                      overlay_trajs=overlay_trajs, overlay_trajs_label=overlay_trajs_lbl,
                      show_legend=show_legend_on_field)
    print(f"Saved results/original_{scenario}.png")

    rows = []

    # ---- DL-only ----
    if RUN_DL:
        print(f"\n=== DL-only training [{scenario}] ===")
        dl = DLOnly(X_u, v_u, X_t0, v_t0, layers, lb, ub, lr=LR_INIT)
        tsec = dl.train(steps=ADAM_STEPS_DL, batch=BATCH_DL, print_every=200, W_T0_val=W_T0)

        v_dl_flat = dl.predict(X_star).reshape(nt, nx)
        relL2 = np.linalg.norm((V - v_dl_flat)[np.isfinite(V)]) / np.linalg.norm(V[np.isfinite(V)])

        rmse_all, mae_all, rmse_cong, mae_cong, rel_cong, cong_frac = compute_velocity_metrics(
            V, v_dl_flat, CONG_SPEED_THRESH
        )

        print(f"[DL/{scenario}] relL2={relL2:.4f} "
              f"RMSE(all)={rmse_all:.4f} MAE(all)={mae_all:.4f} | "
              f"RMSE(v<{CONG_SPEED_THRESH:g})={rmse_cong:.4f} MAE(v<{CONG_SPEED_THRESH:g})={mae_cong:.4f} "
              f"relL2(v<{CONG_SPEED_THRESH:g})={rel_cong:.4f} cong_frac={cong_frac:.3f} time={tsec:.1f}s")

        rows.append(dict(
            model='DL', scenario=scenario,
            Nu=int(X_u.shape[0]), Nf=int(X_f.shape[0]),
            relL2=float(relL2),
            RMSE=float(rmse_all), MAE=float(mae_all),
            RMSE_cong=float(rmse_cong), MAE_cong=float(mae_cong),
            relL2_cong=float(rel_cong),
            cong_frac=float(cong_frac),
            cong_v_thresh=float(CONG_SPEED_THRESH),
            train_sec=float(tsec)
        ))

        plot_field_single(v_dl_flat, x_vec, t_vec, f"DL-only", f"results/v_pred_dl_{scenario}.png",
                          vmin=vmin_data, vmax=vmax_data,
                          overlay_X=overlay_pts, overlay_label=overlay_lbl,
                          overlay_trajs=overlay_trajs, overlay_trajs_label=overlay_trajs_lbl,
                          show_legend=show_legend_on_field)
        print(f"Saved results/v_pred_dl_{scenario}.png")

    # PINN 
    if RUN_PINN:
        print(f"\n=== PINN training [{scenario}] (continuity + Eq.2) ===")
        pinn = PINN_VelRho(
            X_u, v_u, w_u, X_t0, v_t0, X_f, X_ic, rho_ic, layers, lb, ub,
            v_f=v_free,
            rho_m_init=RHO_M_INIT, rho_m_bounds=RHO_M_BOUNDS, fix_rho_m=FIX_RHO_M,
            c0_init=_C0_INIT_USE, c0_bounds=C0_BOUNDS,
            T_init=_T_INIT_USE,  T_bounds=T_BOUNDS,
            use_prior=USE_C0T_PRIOR, prior_c0=_C0_PRIOR_USE, prior_T=_T_PRIOR_USE, prior_w=PRIOR_W,
            w_data=W_DATA, w_cont=W_CONT, w_vel=W_VEL, w_ic=W_IC, w_t0=W_T0,
            lr=LR_INIT
        )

        tsec, hist, ramp_step_used, lr_decay_used = pinn.train(
            steps=ADAM_STEPS_PINN, f_batch=F_BATCH_PINN, d_batch=D_BATCH_PINN,
            print_every=200, lr_decay_steps=LR_DECAY_STEPS
        )

        csv_path = f"results/trainlog_pinn_{scenario}.csv"
        save_hist_csv(hist, csv_path)
        plot_pinn_diagnostics(hist, scenario, ramp_step_used, lr_decay_used)
        print(f"Saved {csv_path}")
        print(f"Saved results/loss_pinn_{scenario}.png")
        print(f"Saved results/trace_params_pinn_{scenario}.png")
        print(f"Saved results/path_c0_T_pinn_{scenario}.png")

        v_pinn_flat = pinn.predict_v(X_star).reshape(nt, nx)
        relL2 = np.linalg.norm((V - v_pinn_flat)[np.isfinite(V)]) / np.linalg.norm(V[np.isfinite(V)])

        rmse_all, mae_all, rmse_cong, mae_cong, rel_cong, cong_frac = compute_velocity_metrics(
            V, v_pinn_flat, CONG_SPEED_THRESH
        )

        print(f"[PINN/{scenario}] relL2={relL2:.4f} "
              f"RMSE(all)={rmse_all:.4f} MAE(all)={mae_all:.4f} | "
              f"RMSE(v<{CONG_SPEED_THRESH:g})={rmse_cong:.4f} MAE(v<{CONG_SPEED_THRESH:g})={mae_cong:.4f} "
              f"relL2(v<{CONG_SPEED_THRESH:g})={rel_cong:.4f} cong_frac={cong_frac:.3f} time={tsec:.1f}s")

        rows.append(dict(
            model='PINN', scenario=scenario,
            Nu=int(X_u.shape[0]), Nf=int(X_f.shape[0]),
            relL2=float(relL2),
            RMSE=float(rmse_all), MAE=float(mae_all),
            RMSE_cong=float(rmse_cong), MAE_cong=float(mae_cong),
            relL2_cong=float(rel_cong),
            cong_frac=float(cong_frac),
            cong_v_thresh=float(CONG_SPEED_THRESH),
            train_sec=float(tsec)
        ))

        plot_field_single(v_pinn_flat, x_vec, t_vec, f"PIDL", f"results/v_pred_pinn_{scenario}.png",
                          vmin=vmin_data, vmax=vmax_data,
                          overlay_X=overlay_pts, overlay_label=overlay_lbl,
                          overlay_trajs=overlay_trajs, overlay_trajs_label=overlay_trajs_lbl,
                          show_legend=show_legend_on_field)
        print(f"Saved results/v_pred_pinn_{scenario}.png")

    if rows:
        keys = sorted({k for r in rows for k in r.keys()})
        with open(f"results/metrics_{scenario}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for r in rows: w.writerow(r)
        print(f"Saved results/metrics_{scenario}.csv")
