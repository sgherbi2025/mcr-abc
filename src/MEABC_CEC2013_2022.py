# %%
import os, sys, time, json, pickle, math, shutil, platform, datetime, warnings, zipfile
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
warnings.filterwarnings("ignore")
sys.path.insert(0, os.getcwd())
from cec2013_functions import CEC2013, F_BIAS as F_BIAS_2013, NAMES as NAMES_2013, TYPES as TYPES_2013, LB as LB_2013, UB as UB_2013
from cec2022_official import CEC2022

QUICK = os.environ.get("MEABC_QUICK", "0") == "1"
SUITES = [s.strip().upper() for s in os.environ.get("MEABC_SUITES", "CEC2013,CEC2022").split(",") if s.strip()]

SN       = 50
LIMIT    = 100
C_GABC   = 1.5
P_MABC   = 0.7
SEED     = 20260101
N_GRID   = 100
BASE_RUNS, BASE_ALGOS, BASE_SAVE_EVERY, BASE_COMPLEXITY = 30, ["MEABC"], 3600, True
ALGO_ID  = {"MEABC": 0, "ABC": 1, "GABC": 2, "MABC": 3}

SUITE_CFG = {
    "CEC2013": dict(D=30, max_fes=lambda d: 10_000 * d, funcs=list(range(1, 29)), quick_funcs=[1, 5, 12, 22], cx_fid=14),
    "CEC2022": dict(D=10, max_fes=lambda d: {10: 200_000}[d], funcs=list(range(1, 13)), quick_funcs=[1, 4, 9, 12], cx_fid=1),
}

NAMES_2022 = {
    1: "Shifted and full Rotated Zakharov Function",
    2: "Shifted and full Rotated Rosenbrock's Function",
    3: "Shifted and full Rotated Schaffer's F7 Function",
    4: "Shifted and full Rotated Non-continuous Rastrigin's Function",
    5: "Shifted and full Rotated Levy Function",
    6: "Hybrid Function 1 (N = 3)",
    7: "Hybrid Function 2 (N = 6)",
    8: "Hybrid Function 3 (N = 5)",
    9: "Composition Function 1 (N = 5)",
    10: "Composition Function 2 (N = 3)",
    11: "Composition Function 3 (N = 5)",
    12: "Composition Function 4 (N = 6)",
}
TYPES_2022 = {1: "Unimodal", **{i: "Basic" for i in range(2, 6)}, **{i: "Hybrid" for i in range(6, 9)}, **{i: "Composition" for i in range(9, 13)}}

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight"})

# %%
def _clip(v):
    return LB if v < LB else (UB if v > UB else v)

def _others(rng, i, n):
    p = rng.choice(SN - 1, n, replace=False)
    return p + (p >= i)

def cycle_meabc(st, ev, rng):
    X, fit, S = st["X"], st["fit"], st["S"]
    gb = X[np.argmin(fit)].copy()
    for i in range(SN):
        if st["fes"] >= MAX_FES:
            break
        j = int(rng.integers(D)); phi = rng.uniform(-1, 1)
        k = int(rng.integers(SN - 1)); k += (k >= i)
        xij = X[i, j]; s = S[i]
        if s == 0:
            v = xij + phi * (xij - X[k, j])
        elif s == 1:
            v = xij + phi * (xij - X[k, j]) + rng.uniform(0, C_GABC) * (gb[j] - xij)
        else:
            v = gb[j] + phi * (gb[j] - X[k, j])
        y = X[i].copy(); y[j] = _clip(v)
        fy = ev(y); st["fes"] += 1
        if fy < fit[i]:
            X[i] = y; fit[i] = fy
        else:
            S[i] = (S[i] + 1 + int(rng.integers(2))) % 3

def cycle_abc(st, ev, rng, guided):
    X, fit, trial = st["X"], st["fit"], st["trial"]
    gb = X[np.argmin(fit)].copy()
    def attempt(i):
        j = int(rng.integers(D)); phi = rng.uniform(-1, 1)
        k = int(rng.integers(SN - 1)); k += (k >= i)
        v = X[i, j] + phi * (X[i, j] - X[k, j])
        if guided:
            v += rng.uniform(0, C_GABC) * (gb[j] - X[i, j])
        y = X[i].copy(); y[j] = _clip(v)
        fy = ev(y); st["fes"] += 1
        if fy < fit[i]:
            X[i] = y; fit[i] = fy; trial[i] = 0
        else:
            trial[i] += 1
    for i in range(SN):
        if st["fes"] >= MAX_FES: return
        attempt(i)
    q = 1.0 / (1.0 + np.maximum(fit, 0.0)); cum = np.cumsum(q)
    for _ in range(SN):
        if st["fes"] >= MAX_FES: return
        i = min(int(np.searchsorted(cum, rng.random() * cum[-1])), SN - 1)
        attempt(i)
    m = int(np.argmax(trial))
    if trial[m] > LIMIT and st["fes"] < MAX_FES:
        X[m] = rng.uniform(LB, UB, D); fit[m] = ev(X[m]); st["fes"] += 1; trial[m] = 0

def cycle_mabc(st, ev, rng):
    X, fit = st["X"], st["fit"]
    gb = X[np.argmin(fit)].copy()
    for i in range(SN):
        if st["fes"] >= MAX_FES: break
        j = int(rng.integers(D)); phi = rng.uniform(-1, 1)
        if rng.random() < P_MABC:
            k = _others(rng, i, 1)[0]; v = X[i, j] + phi * (X[i, j] - X[k, j])
        else:
            r1, r2 = _others(rng, i, 2); v = gb[j] + phi * (X[r1, j] - X[r2, j])
        y = X[i].copy(); y[j] = _clip(v)
        fy = ev(y); st["fes"] += 1
        if fy < fit[i]:
            X[i] = y; fit[i] = fy

def _record(st):
    b = float(np.min(st["fit"]))
    while st["ng"] <= N_GRID and st["fes"] >= GRID[st["ng"]]:
        st["hist"].append(b); st["ng"] += 1

def run_algo(algo, f, fid, rng, state=None, cb=None):
    fb = bias_of(fid)
    ev = lambda x: float(f(x)) - fb
    if state is None:
        X = rng.uniform(LB, UB, (SN, D))
        fit = np.array([ev(x) for x in X])
        state = dict(algo=algo, fid=fid, X=X, fit=fit, S=rng.integers(0, 3, SN),
                     trial=np.zeros(SN, dtype=int), fes=SN, hist=[], ng=0, elapsed=0.0)
        _record(state)
    else:
        rng.bit_generator.state = state["rng_state"]
    t_last = time.perf_counter()
    while state["fes"] < MAX_FES:
        if algo == "MEABC":  cycle_meabc(state, ev, rng)
        elif algo == "ABC":  cycle_abc(state, ev, rng, False)
        elif algo == "GABC": cycle_abc(state, ev, rng, True)
        elif algo == "MABC": cycle_mabc(state, ev, rng)
        _record(state)
        now = time.perf_counter(); state["elapsed"] += now - t_last
        if cb is not None: cb(state)
        t_last = time.perf_counter()
    while len(state["hist"]) < N_GRID + 1:
        state["hist"].append(float(np.min(state["fit"])))
    return dict(best=float(np.min(state["fit"])), time=state["elapsed"], fes=state["fes"],
                hist=np.array(state["hist"]))

# %%
class Checkpointer:
    def __init__(self, out, every):
        self.out, self.every = out, every
        self.res, self.partial = {}, None
        self.latest = os.path.join(out, "checkpoint_latest.pkl")
        self.snapdir = os.path.join(out, "checkpoints_hourly"); os.makedirs(self.snapdir, exist_ok=True)
        self.logf = os.path.join(out, "run_log.txt")
        self.last = time.time()
        if os.path.exists(self.latest):
            with open(self.latest, "rb") as fh: d = pickle.load(fh)
            if d["proto"] != PROTO:
                raise RuntimeError("Un checkpoint existe avec un autre protocole : changez OUT ou supprimez %s" % self.latest)
            self.res, self.partial = d["res"], d["partial"]
            self.log("REPRISE : %d runs déjà terminés%s" % (len(self.res), " + 1 run partiel" if self.partial else ""))
    def log(self, msg):
        line = "[%s] %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), msg)
        print(line)
        with open(self.logf, "a", encoding="utf-8") as fh: fh.write(line + "\n")
    def _dump(self, path):
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(dict(res=self.res, partial=self.partial, proto=PROTO,
                             saved=datetime.datetime.now().isoformat()), fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    def save(self, hourly=False):
        self._dump(self.latest)
        if hourly:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            shutil.copy(self.latest, os.path.join(self.snapdir, "ckpt_%s.pkl" % ts))
            rows = [dict(algo=k[0], func=k[1], run=k[2], error=v["best"], time_s=v["time"]) for k, v in self.res.items()]
            pd.DataFrame(rows).to_csv(os.path.join(self.out, "partial_results_%s.csv" % ts), index=False)
            self.log("SAUVEGARDE HORAIRE : %d runs terminés (snapshot ckpt_%s.pkl)" % (len(self.res), ts))
    def tick(self, key, st, rng):
        if time.time() - self.last >= self.every:
            st["rng_state"] = rng.bit_generator.state
            self.partial = (key, st)
            self.save(hourly=True)
            self.last = time.time()
    def take_partial(self, key):
        if self.partial is not None and self.partial[0] == key:
            return self.partial[1]
        return None
    def done(self, key, r):
        self.res[key] = r; self.partial = None; self.save(hourly=False)

# %%
_FCACHE = {}

def get_f(fid):
    if fid not in _FCACHE:
        _FCACHE[fid] = CEC2013(fid, D) if SUITE == "CEC2013" else CEC2022(fid, D)
    return _FCACHE[fid]

def setup(suite):
    global SUITE, D, MAX_FES, FUNCS, RUNS, ALGOS, SAVE_EVERY, RUN_COMPLEXITY, OUT, PROTO, GRID
    global LB, UB, NAMES, TYPES, CX_FID, bias_of, CK, _FCACHE
    cfg = SUITE_CFG[suite]
    SUITE = suite
    D = cfg["D"]; MAX_FES = cfg["max_fes"](D); FUNCS = list(cfg["funcs"]); CX_FID = cfg["cx_fid"]
    RUNS, ALGOS, SAVE_EVERY, RUN_COMPLEXITY = BASE_RUNS, list(BASE_ALGOS), BASE_SAVE_EVERY, BASE_COMPLEXITY
    OUT = "results_meabc_%s_D%d" % (suite.lower(), D)
    if QUICK:
        RUNS, MAX_FES, FUNCS, SAVE_EVERY = 3, 3000, list(cfg["quick_funcs"]), 3
        OUT = "results_quick_%s" % suite.lower()
        ALGOS, RUN_COMPLEXITY = ["MEABC", "ABC"], False
    _FCACHE = {}
    if suite == "CEC2013":
        LB, UB, NAMES, TYPES = LB_2013, UB_2013, NAMES_2013, TYPES_2013
        bias_of = lambda fid: F_BIAS_2013[fid]
    else:
        LB, UB, NAMES, TYPES = -100.0, 100.0, NAMES_2022, TYPES_2022
        bias_of = lambda fid: get_f(fid).f_opt
    GRID = np.linspace(0, MAX_FES, N_GRID + 1).astype(int)
    PROTO = dict(D=D, SN=SN, MAX_FES=MAX_FES, LIMIT=LIMIT, C=C_GABC, P=P_MABC, SEED=SEED)
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    print("Suite :", suite, "| Protocole :", PROTO, "| algos :", ALGOS, "| QUICK =", QUICK)
    CK = Checkpointer(OUT, SAVE_EVERY)

def estimate_runtime(n_eval=200, overhead_us=30.0):
    tot = 0.0; rows = []
    for fid in FUNCS:
        f = get_f(fid); X = np.random.default_rng(0).uniform(LB, UB, (n_eval, D))
        t = time.perf_counter()
        for x in X: f(x)
        us = (time.perf_counter() - t) / n_eval * 1e6
        rows.append((fid, us)); tot += (us + overhead_us) * 1e-6 * MAX_FES * RUNS
    tot *= len(ALGOS)
    print("Coût moyen d'une évaluation par fonction (µs) :")
    print({fid: round(us, 1) for fid, us in rows})
    print("Estimation séquentielle totale : %.1f h (%.1f jours) pour %s" % (tot / 3600, tot / 86400, ALGOS))

def campaign():
    todo = [(a, fid, r) for a in ALGOS for fid in FUNCS for r in range(RUNS)]
    n_tot = len(todo); t0 = time.time(); done_time = 0.0; n_new = 0
    CK.log("DÉBUT/REPRISE campagne %s : %d runs au total, %d déjà faits" % (SUITE, n_tot, len(CK.res)))
    try:
        for (a, fid, r) in todo:
            key = (a, fid, r)
            if key in CK.res: continue
            rng = np.random.default_rng(np.random.SeedSequence([SEED, fid, r, ALGO_ID[a]]))
            st = CK.take_partial(key)
            res = run_algo(a, get_f(fid), fid, rng, state=st, cb=lambda s, k=key, g=rng: CK.tick(k, s, g))
            CK.done(key, res); n_new += 1; done_time += res["time"]
            left = n_tot - len(CK.res)
            eta = left * done_time / max(n_new, 1)
            err = 0.0 if res["best"] < 1e-8 else res["best"]
            CK.log("%-6s F%-2d run %2d/%d | erreur=%.3E | %.1fs | restants=%d | ETA≈%.2f h" %
                   (a, fid, r + 1, RUNS, err, res["time"], left, eta / 3600))
    except KeyboardInterrupt:
        CK.log("INTERRUPTION : sauvegarde finale"); CK.save(hourly=True); raise
    CK.save(hourly=True)
    CK.log("CAMPAGNE %s TERMINÉE en %.2f h (wall, cette session)" % (SUITE, (time.time() - t0) / 3600))

# %%
def build_df():
    rows = []
    for (a, fid, r), v in CK.res.items():
        e = v["best"]; e = 0.0 if abs(e) < 1e-8 else e
        rows.append(dict(algo=a, func=fid, run=r, error=e, time_s=v["time"], fes=v["fes"]))
    return pd.DataFrame(rows).sort_values(["algo", "func", "run"]).reset_index(drop=True)

def stats_table(df, algo):
    rows = []
    for fid in FUNCS:
        d = df[(df.algo == algo) & (df.func == fid)]
        if len(d) == 0: continue
        e = d.error.values
        rows.append(dict(F="F%d" % fid, Name=NAMES[fid], Type=TYPES[fid], Best=e.min(), Worst=e.max(),
                         Median=np.median(e), Mean=e.mean(), Std=e.std(ddof=1) if len(e) > 1 else 0.0,
                         Time_mean_s=d.time_s.mean(), Time_std_s=d.time_s.std(ddof=1) if len(d) > 1 else 0.0, n=len(d)))
    return pd.DataFrame(rows)

def fE(x):
    return "0" if x == 0 else ("%.2E" % x).replace("E", r"\mathrm{E}")

def latex_stats(t, algo):
    L = [r"\begin{table*}[htbp]", r"\centering", r"\scriptsize",
         r"\caption{Erreurs de %s sur les fonctions %s ($D=%d$, %d runs, $MaxFEs=%d$). Erreurs $<10^{-8}$ ramenées à 0.}" % (algo, SUITE.replace("CEC", "CEC "), D, RUNS, MAX_FES),
         r"\label{tab:%s_%s}" % (algo.lower(), SUITE.lower()), r"\begin{tabular}{llrrrrrr}", r"\toprule",
         r"F & Fonction & Best & Worst & Median & Mean & Std & Temps (s) \\", r"\midrule"]
    for _, r in t.iterrows():
        L.append(r"%s & %s & $%s$ & $%s$ & $%s$ & $%s$ & $%s$ & %.1f \\" %
                 (r.F, r.Name.replace("&", r"\&"), fE(r.Best), fE(r.Worst), fE(r.Median), fE(r.Mean), fE(r.Std), r.Time_mean_s))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)

def export_tables(df):
    df.to_csv(os.path.join(OUT, "all_runs.csv"), index=False)
    TABLES = {a: stats_table(df, a) for a in ALGOS}
    for a, t in TABLES.items():
        t.to_csv(os.path.join(OUT, "tables", "stats_%s.csv" % a), index=False)
        print("=== %s | %s ===" % (SUITE, a))
        print(t.drop(columns=["Name"]).to_string(index=False, float_format=lambda x: "%.3E" % x))
        print("Temps total CPU séquentiel (somme des runs) : %.2f h | moyenne par run : %.1f s" %
              (df[df.algo == a].time_s.sum() / 3600, df[df.algo == a].time_s.mean()))
    tt = df.groupby(["algo", "func"]).time_s.agg(["mean", "std", "sum"]).reset_index()
    tt.to_csv(os.path.join(OUT, "tables", "execution_time.csv"), index=False)
    try:
        with pd.ExcelWriter(os.path.join(OUT, "results_summary.xlsx")) as xw:
            df.to_excel(xw, sheet_name="all_runs", index=False)
            for a, t in TABLES.items(): t.to_excel(xw, sheet_name="stats_%s" % a, index=False)
            tt.to_excel(xw, sheet_name="time", index=False)
    except Exception as ex:
        print("Export Excel ignoré :", ex)
    for a, t in TABLES.items():
        p = os.path.join(OUT, "tables", "table_%s.tex" % a)
        with open(p, "w", encoding="utf-8") as fh: fh.write(latex_stats(t, a))
        print("LaTeX écrit :", p)
    return TABLES, tt

# %%
LIT = {
 "DE-APC":       [0,1.75e5,3.21e6,2.21e-1,0,9.35,2.18e1,2.09e1,3.07e1,6.42e-2,3.08,3.17e1,7.55e1,3.84e3,4.14e3,2.46,5.92e1,6.04e1,2.30,1.26e1,2.67e2,4.56e3,4.18e3,2.92e2,2.99e2,3.29e2,1.19e3,3.00e2],
 "fk-PSO":       [0,1.59e6,2.40e8,4.78e2,0,2.99e1,6.39e1,2.09e1,1.85e1,2.29e-1,2.36e1,5.64e1,1.23e2,7.04e2,3.42e3,8.48e-1,5.26e1,6.81e1,3.12,1.20e1,3.11e2,8.59e2,3.57e3,2.48e2,2.49e2,2.95e2,7.76e2,4.01e2],
 "ADE":          [0,2.18e6,1.65e3,1.70e4,1.40e-7,8.29,1.29,2.09e1,6.30,2.16e-2,5.84e1,1.15e2,1.31e2,3.20e3,5.61e3,2.39,1.02e2,1.82e2,5.40,1.13e1,3.19e2,2.50e3,5.81e3,2.02e2,2.30e2,2.18e2,3.26e2,3.00e2],
 "MEABC (article)": [0,1.23e6,1.40e8,8.35e4,0,1.01e1,9.23e1,2.09e1,2.88e1,5.57,0,2.07e2,2.29e2,1.37e1,3.41e2,1.44,3.04e1,1.80e2,3.94e-1,1.56e1,2.10e2,1.78e1,5.16e3,2.81e2,2.74e2,2.01e2,4.02e2,3.00e2],
}
LIT = {k: np.array([0.0 if v < 1e-8 else v for v in vals]) for k, vals in LIT.items()}

def same3(a, b):
    f = lambda x: 0.0 if x == 0 else float("%.3g" % x)
    return f(a) == f(b)

def wtl(a, b):
    w = t = l = 0
    for x, y in zip(a, b):
        if same3(x, y): t += 1
        elif x < y: w += 1
        else: l += 1
    return w, t, l

def signed_rank(a, b):
    d = np.asarray(a) - np.asarray(b); nz = d != 0
    if nz.sum() == 0: return 0.0, 0.0, 1.0
    rk = stats.rankdata(np.abs(d[nz]))
    Rp = rk[d[nz] < 0].sum(); Rm = rk[d[nz] > 0].sum()
    try: p = stats.wilcoxon(a, b, zero_method="zsplit").pvalue
    except Exception: p = 1.0
    return Rp, Rm, p

def literature(TABLES):
    fid_list = [f for f in FUNCS]
    idx = [f - 1 for f in fid_list]
    ours = {}
    for a in ALGOS:
        t = TABLES[a].set_index("F")
        ours[a] = np.array([t.loc["F%d" % f, "Mean"] for f in fid_list if "F%d" % f in t.index])
    main = ALGOS[0]
    cmp_rows = []
    for k, v in LIT.items():
        w, t_, l = wtl(ours[main], v[idx]); Rp, Rm, p = signed_rank(ours[main], v[idx])
        cmp_rows.append(dict(Comparé_à=k, Wins=w, Ties=t_, Losses=l, R_plus=Rp, R_minus=Rm, p_Wilcoxon=p,
                             Signif_5pct="oui" if p < 0.05 else "non"))
    cmp = pd.DataFrame(cmp_rows); cmp.to_csv(os.path.join(OUT, "tables", "wilcoxon_vs_literature.csv"), index=False)
    print("== %s (cette étude) vs résultats publiés ==" % main); print(cmp.to_string(index=False))
    names = [main + " (cette étude)"] + list(LIT.keys())
    M = np.column_stack([ours[main]] + [LIT[k][idx] for k in LIT])
    ranks = np.apply_along_axis(stats.rankdata, 1, M)
    avg = ranks.mean(axis=0)
    chi, pf = stats.friedmanchisquare(*[M[:, i] for i in range(M.shape[1])])
    fr = pd.DataFrame(dict(Algorithme=names, Rang_moyen=avg)).sort_values("Rang_moyen")
    fr.to_csv(os.path.join(OUT, "tables", "friedman_ranks.csv"), index=False)
    print("\nFriedman : chi2=%.3f, p=%.4g" % (chi, pf)); print(fr.to_string(index=False))
    N, kk = M.shape
    se = math.sqrt(kk * (kk + 1) / (6.0 * N)); ctl = int(np.argmin(avg))
    ph = [(names[i], (avg[i] - avg[ctl]) / se) for i in range(kk) if i != ctl]
    ph = [(n, z, 2 * (1 - stats.norm.cdf(abs(z)))) for n, z in ph]
    ph.sort(key=lambda r: r[2]); holm = []
    for i, (n, z, p) in enumerate(ph):
        holm.append(dict(Contrôle=names[ctl], Comparé=n, z=z, p=p, p_Holm=min(1.0, max([q[2] * (len(ph) - j) for j, q in enumerate(ph[:i + 1])]))))
    holm = pd.DataFrame(holm); holm.to_csv(os.path.join(OUT, "tables", "holm_posthoc.csv"), index=False)
    print("\nPost-hoc de Holm (contrôle = meilleur rang) :"); print(holm.to_string(index=False))
    L = [r"\begin{table*}[htbp]", r"\centering", r"\scriptsize", r"\caption{Erreur moyenne sur CEC 2013 ($D=30$) : %s vs algorithmes de la littérature (Table 11).}" % main,
         r"\label{tab:comparison_cec2013}", r"\begin{tabular}{l" + "r" * (len(names)) + "}", r"\toprule", "F & " + " & ".join(names) + r" \\", r"\midrule"]
    for i, f in enumerate(fid_list):
        row = M[i]; best = row.min()
        cells_ = [("$\\mathbf{%s}$" if same3(x, best) else "$%s$") % fE(x) for x in row]
        L.append("F%d & " % f + " & ".join(cells_) + r" \\")
    L += [r"\midrule", "Rang moyen & " + " & ".join("%.2f" % x for x in avg) + r" \\", r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    with open(os.path.join(OUT, "tables", "table_comparison.tex"), "w", encoding="utf-8") as fh: fh.write("\n".join(L))
    return fr

def compare_runs(df, TABLES):
    main = ALGOS[0]
    if len(ALGOS) > 1:
        rows = []
        for b in ALGOS[1:]:
            plus = minus = eq = 0
            for fid in FUNCS:
                x = df[(df.algo == main) & (df.func == fid)].error.values
                y = df[(df.algo == b) & (df.func == fid)].error.values
                if len(x) < 2 or len(y) < 2 or (np.all(x == 0) and np.all(y == 0)): eq += 1; continue
                p = stats.mannwhitneyu(x, y, alternative="two-sided").pvalue
                if p >= 0.05: eq += 1
                elif np.median(x) < np.median(y): plus += 1
                else: minus += 1
            rows.append(dict(Comparé_à=b, **{"+ (MEABC meilleur)": plus, "≈": eq, "− (MEABC moins bon)": minus}))
        pr = pd.DataFrame(rows); pr.to_csv(os.path.join(OUT, "tables", "ranksum_runs.csv"), index=False)
        print("Test de Mann-Whitney (rank-sum) par fonction, alpha=0.05 :"); print(pr.to_string(index=False))
        mm = np.column_stack([[TABLES[a].set_index("F").loc["F%d" % f, "Mean"] for f in FUNCS] for a in ALGOS])
        rk = np.apply_along_axis(stats.rankdata, 1, mm).mean(axis=0)
        print("Rangs moyens (Friedman) :", dict(zip(ALGOS, np.round(rk, 3))))
    else:
        print("Un seul algorithme exécuté : comparaison par runs ignorée (ajouter 'ABC','GABC','MABC' à BASE_ALGOS).")

# %%
COL = {"MEABC": "#d62728", "ABC": "#1f77b4", "GABC": "#2ca02c", "MABC": "#9467bd"}

def save_fig(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, "figures", "%s.%s" % (name, ext)))

def figures(df, tt, fr):
    nF = len(FUNCS); ncol = 4; nrow = int(math.ceil(nF / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.5 * nrow), squeeze=False)
    conv_rows = []
    for ax, fid in zip(axes.ravel(), FUNCS):
        for a in ALGOS:
            H = np.array([CK.res[(a, fid, r)]["hist"] for r in range(RUNS) if (a, fid, r) in CK.res])
            if len(H) == 0: continue
            H = np.where(H < 1e-8, 1e-8, H); x = GRID / MAX_FES * MAX_FES
            med = np.median(H, 0); q1, q3 = np.percentile(H, 25, 0), np.percentile(H, 75, 0)
            ax.plot(x, med, color=COL.get(a), lw=1.3, label=a); ax.fill_between(x, q1, q3, color=COL.get(a), alpha=0.15)
            conv_rows += [dict(algo=a, func=fid, FEs=int(g), median_error=m) for g, m in zip(GRID, med)]
        ax.set_yscale("log"); ax.set_title("F%d %s" % (fid, NAMES[fid].replace(" Function", "")[:28]), fontsize=7)
        ax.tick_params(labelsize=6)
    for ax in axes.ravel()[nF:]: ax.axis("off")
    axes[0, 0].legend(fontsize=6); fig.supxlabel("Évaluations de fonction (FEs)"); fig.supylabel("Erreur (médiane, IQR)")
    fig.tight_layout(); save_fig(fig, "convergence_all"); plt.show()
    pd.DataFrame(conv_rows).to_csv(os.path.join(OUT, "tables", "convergence_median.csv"), index=False)

    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.5 * nrow), squeeze=False)
    for ax, fid in zip(axes.ravel(), FUNCS):
        data = [np.maximum(df[(df.algo == a) & (df.func == fid)].error.values, 1e-8) for a in ALGOS]
        bp = ax.boxplot(data, patch_artist=True, showfliers=True, flierprops=dict(markersize=2))
        ax.set_xticks(range(1, len(ALGOS)+1)); ax.set_xticklabels(ALGOS)
        for patch, a in zip(bp["boxes"], ALGOS): patch.set_facecolor(COL.get(a)); patch.set_alpha(0.5)
        ax.set_yscale("log"); ax.set_title("F%d" % fid, fontsize=8); ax.tick_params(labelsize=6)
    for ax in axes.ravel()[nF:]: ax.axis("off")
    fig.suptitle("Distribution des erreurs finales (%d runs)" % RUNS, fontsize=10); fig.tight_layout()
    save_fig(fig, "boxplots_all"); plt.show()

    fig, ax = plt.subplots(figsize=(9, 3.2)); w = 0.8 / len(ALGOS)
    for i, a in enumerate(ALGOS):
        t = tt[tt.algo == a].set_index("func")
        ax.bar(np.arange(len(t)) + i * w, t["mean"], w, yerr=t["std"].fillna(0), color=COL.get(a), alpha=0.8, label=a, capsize=1.5)
    ax.set_xticks(np.arange(len(FUNCS)) + w * (len(ALGOS) - 1) / 2); ax.set_xticklabels(["F%d" % f for f in FUNCS], rotation=90, fontsize=7)
    ax.set_ylabel("Temps par run (s)"); ax.legend(); fig.tight_layout(); save_fig(fig, "execution_time"); plt.show()

    if fr is not None:
        fig, ax = plt.subplots(figsize=(5.5, 2.8)); fs = fr.sort_values("Rang_moyen", ascending=False)
        ax.barh(fs.Algorithme, fs.Rang_moyen, color="#7f7f7f"); ax.set_xlabel("Rang moyen de Friedman (28 fonctions)")
        fig.tight_layout(); save_fig(fig, "friedman_ranks"); plt.show()

# %%
def complexity(n_t2_runs=5):
    global MAX_FES
    t = time.perf_counter(); x = 0.55
    for _ in range(1_000_000):
        x = 0.55; x = x + x; x = x / 2; x = x * x; x = math.sqrt(x); x = math.log(x); x = math.exp(x); x = x / (x + 2)
    T0 = time.perf_counter() - t
    f = get_f(CX_FID); X = np.random.default_rng(1).uniform(LB, UB, (200000, D))
    t = time.perf_counter()
    for r in X: f(r)
    T1 = time.perf_counter() - t
    save = MAX_FES; MAX_FES = 200000
    T2s = []
    try:
        for r in range(n_t2_runs):
            t = time.perf_counter()
            run_algo("MEABC", f, CX_FID, np.random.default_rng(777 + r))
            T2s.append(time.perf_counter() - t)
    finally:
        MAX_FES = save
    T2 = float(np.mean(T2s))
    return dict(T0=T0, T1=T1, T2=T2, complexity=(T2 - T1) / T0)

def finalize(df):
    import scipy
    info = dict(suite=SUITE, protocol=PROTO, algos=ALGOS, funcs=FUNCS, runs=RUNS, python=sys.version, numpy=np.__version__,
                scipy=scipy.__version__, pandas=pd.__version__, platform=platform.platform(), processor=platform.processor(),
                parallelism="aucun (exécution séquentielle)", n_runs_done=len(CK.res),
                total_runtime_hours={a: float(df[df.algo == a].time_s.sum() / 3600) for a in ALGOS},
                date=datetime.datetime.now().isoformat())
    with open(os.path.join(OUT, "environment_and_protocol.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2, ensure_ascii=False, default=str)
    zp = OUT + ".zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(OUT):
            if "checkpoints_hourly" in root: continue
            for fn in files: z.write(os.path.join(root, fn))
    print("Archive des résultats :", zp)
    print(json.dumps(info["total_runtime_hours"], indent=1))

def report():
    df = build_df()
    TABLES, tt = export_tables(df)
    fr = literature(TABLES) if SUITE == "CEC2013" else None
    compare_runs(df, TABLES)
    figures(df, tt, fr)
    if RUN_COMPLEXITY:
        cx = complexity()
        print(cx); pd.DataFrame([cx]).to_csv(os.path.join(OUT, "tables", "complexity_T0_T1_T2.csv"), index=False)
    else:
        print("Complexité ignorée (RUN_COMPLEXITY = False)")
    finalize(df)

# %%
for _suite in SUITES:
    if _suite not in SUITE_CFG:
        raise ValueError("Suite inconnue : %s (choix : %s)" % (_suite, list(SUITE_CFG)))
    setup(_suite)
    estimate_runtime()
    campaign()
    report()
