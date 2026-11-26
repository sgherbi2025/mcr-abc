# %%
import os, sys, json, time, math, platform, datetime, zipfile
from types import SimpleNamespace
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

SN        = 50
LIMIT     = 100
RUNS      = 30
EXPERIMENTS = [("CEC2022", 10), ("CEC2013", 30)]

K_INIT  = (SN - 1) // 2
PSI_MAX = 1.5
EPS_DEN = 1e-12

BASE_SEED          = 20260121
CHECKPOINT_EVERY_S = 3600
N_CONV_POINTS      = 100
STD_DDOF           = 1
ALGO_NAME          = "ABCNG"

QUICK_TEST  = False
QUICK_FUNCS = {"CEC2022": [1, 5, 7, 9, 12], "CEC2013": [1, 5, 8, 12, 15, 21, 28]}

DISPLAY = {"CEC2022": "CEC 2022", "CEC2013": "CEC 2013"}


def max_fes_for(bench, dim):
    if bench == "CEC2013":
        return 10_000 * dim
    return {10: 200_000, 20: 1_000_000}[dim]


# %%
NAMES_2022 = {1: "Shifted and Full Rotated Zakharov", 2: "Shifted and Full Rotated Rosenbrock",
              3: "Shifted and Full Rotated Expanded Schaffer f6", 4: "Shifted and Full Rotated Non-continuous Rastrigin",
              5: "Shifted and Full Rotated Levy", 6: "Hybrid Function 1 (N=3)", 7: "Hybrid Function 2 (N=6)",
              8: "Hybrid Function 3 (N=5)", 9: "Composition Function 1 (N=5)", 10: "Composition Function 2 (N=4)",
              11: "Composition Function 3 (N=5)", 12: "Composition Function 4 (N=6)"}
TYPES_2022 = {1: "Unimodal", **{i: "Basic" for i in range(2, 6)}, **{i: "Hybrid" for i in range(6, 9)},
              **{i: "Composition" for i in range(9, 13)}}


def _prefer_path(module_file):
    for p in ("/mnt/user-data/uploads", os.getcwd()):
        if os.path.exists(os.path.join(p, module_file)):
            if p in sys.path: sys.path.remove(p)
            sys.path.insert(0, p)


def load_benchmark(bench, dim):
    if bench == "CEC2022":
        _prefer_path("cec2022_official.py")
        from cec2022_official import CEC2022
        dd = os.path.join(os.path.dirname(os.path.abspath(sys.modules["cec2022_official"].__file__)), "input_data")
        if not os.path.isdir(dd):
            raise FileNotFoundError(f"Dossier des données officielles CEC2022 introuvable : {dd}")
        print("cec2022_official importé :", sys.modules["cec2022_official"].__file__)
        return (lambda fid: CEC2022(fid, dim)), NAMES_2022, TYPES_2022, list(range(1, 13))
    if bench == "CEC2013":
        _prefer_path("cec2013_functions.py")
        import cec2013_functions as c13
        if dim not in c13.DIMS:
            raise ValueError(f"CEC2013 : dimension {dim} non disponible ({c13.DIMS})")
        print("cec2013_functions importé :", sys.modules["cec2013_functions"].__file__)
        return (lambda fid: c13.CEC2013(fid, dim)), c13.NAMES, c13.TYPES, list(c13.FUNC_IDS)
    raise ValueError(f"Benchmark inconnu : {bench}")


def build_cfg(bench, dim):
    make, names, types, func_ids = load_benchmark(bench, dim)
    max_fes, runs = max_fes_for(bench, dim), RUNS
    out = f"./abcng_results_{bench.lower()}_{dim}"
    if QUICK_TEST:
        max_fes, runs, func_ids = 3_000, 2, QUICK_FUNCS[bench]
        out += "_QUICKTEST"
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(out, "figures", "convergence"), exist_ok=True)
    cfg = SimpleNamespace(bench=bench, D=dim, make=make, names=names, types=types, func_ids=func_ids,
                          max_fes=max_fes, runs=runs, out=out, fig_dir=os.path.join(out, "figures"),
                          tag=f"{ALGO_NAME.lower()}_{bench.lower()}_D{dim}")
    print(f"{bench}  D={dim}  SN={SN}  MaxFEs={max_fes}  LIMIT={LIMIT}  RUNS={runs}  fonctions={len(func_ids)}")
    return cfg


# %%
def conv_marks(max_fes, n_conv=N_CONV_POINTS):
    return np.round(np.linspace(max_fes / n_conv, max_fes, n_conv)).astype(np.int64)


def run_abcng(fobj, dim, lb, ub, f_opt, sn, limit, max_fes, rng,
              n_conv=N_CONV_POINTS, k_init=K_INIT, psi_max=PSI_MAX):
    ev   = getattr(fobj, "_eval", fobj)
    lb   = np.asarray(lb, float); ub = np.asarray(ub, float); span = ub - lb
    kmax = (sn - 1) // 2
    marks = conv_marks(max_fes, n_conv)
    hist  = np.empty(n_conv)
    fes, gf, gbest, nxt = 0, np.inf, None, 0

    def evaluate(x):
        nonlocal fes, gf, gbest, nxt
        fx = float(ev(x)); fes += 1
        if fx < gf:
            gf, gbest = fx, x.copy()
        while nxt < n_conv and fes >= marks[nxt]:
            hist[nxt] = gf - f_opt; nxt += 1
        return fx

    X = lb + rng.random((sn, dim)) * span
    F = np.array([evaluate(X[i]) for i in range(sn)])
    trial   = np.zeros(sn, dtype=np.int64)
    k       = k_init
    delta_i = np.zeros(sn)
    delta_a = 0.0

    def propose(i):
        j   = int(rng.integers(dim))
        off = int(rng.integers(1, k + 1))
        if rng.random() < 0.5: off = -off
        ni  = (i + off) % sn
        n_out = sn - 2 * k - 1
        if n_out > 0:
            no = (i + k + 1 + int(rng.integers(n_out))) % sn
        else:
            no = int(rng.integers(sn))
            while no == ni or no == i: no = int(rng.integers(sn))
        xn = X[ni, j]
        vj = xn + rng.uniform(-1.0, 1.0) * (xn - X[no, j]) + rng.uniform(0.0, psi_max) * (gbest[j] - xn)
        v = X[i].copy()
        v[j] = min(max(vj, lb[j]), ub[j])
        return v

    def try_replace(i, v):
        fv = evaluate(v)
        if fv < F[i]:
            X[i] = v; F[i] = fv; trial[i] = 0
            return True
        return False

    while fes < max_fes:
        for i in range(sn):
            if fes >= max_fes: break
            if try_replace(i, propose(i)): k = min(k + 1, kmax)
            else:                          trial[i] += 1; k = max(k - 1, 1)
        if fes >= max_fes: break

        F_prev = F.copy()
        for i in range(sn):
            if fes >= max_fes: break
            if try_replace(i, propose(i)):
                k = min(k + 1, kmax); continue
            k = max(k - 1, 1)
            if fes >= max_fes:
                trial[i] += 1; break
            mu, sd = delta_i[i], delta_a
            if mu == 0.0 and sd == 0.0:
                trial[i] += 1; continue
            v = X[i] * (1.0 + rng.normal(mu, sd, dim))
            np.clip(v, lb, ub, out=v)
            if not try_replace(i, v): trial[i] += 1

        den     = np.where(np.abs(F) < EPS_DEN, EPS_DEN, F)
        delta_i = np.nan_to_num((F - F_prev) / den, nan=0.0, posinf=0.0, neginf=0.0)
        delta_a = abs(float(delta_i.mean()))

        if fes < max_fes:
            m = int(np.argmax(trial))
            if trial[m] > limit:
                X[m] = lb + rng.random(dim) * span
                F[m] = evaluate(X[m]); trial[m] = 0; delta_i[m] = 0.0

    while nxt < n_conv:
        hist[nxt] = gf - f_opt; nxt += 1
    return dict(best_err=gf - f_opt, best_f=gf, best_x=gbest, fes=fes, hist=hist, final_k=k)


# %%
for _bench, _dim in EXPERIMENTS:
    _make, _, _, _ = load_benchmark(_bench, _dim)
    _f = _make(1)
    _t = time.perf_counter()
    with np.errstate(all="ignore"):
        _r = run_abcng(_f, _dim, _f.lb, _f.ub, _f.f_opt, SN, LIMIT, 20_000,
                       np.random.default_rng(np.random.SeedSequence([BASE_SEED, 1, 0])))
    print(f"{_bench} D={_dim} F1, 20 000 FEs -> erreur = {_r['best_err']:.3e} | FEs = {_r['fes']} | {time.perf_counter()-_t:.1f} s")
    del _make, _f, _r, _t

# %%
def execute(cfg):
    ckpt = os.path.join(cfg.out, "checkpoint.json")
    config = dict(algorithm=ALGO_NAME, benchmark=cfg.bench, D=cfg.D, SN=SN, MaxFEs=cfg.max_fes, LIMIT=LIMIT,
                  RUNS=cfg.runs, funcs=cfg.func_ids, K_INIT=K_INIT, PSI_MAX=PSI_MAX, EPS_DEN=EPS_DEN,
                  BASE_SEED=BASE_SEED, N_CONV_POINTS=N_CONV_POINTS, STD_DDOF=STD_DDOF,
                  threshold_1e8_applied=False, parallelism=False)
    keys = ["benchmark", "D", "SN", "MaxFEs", "LIMIT", "K_INIT", "PSI_MAX", "EPS_DEN", "BASE_SEED", "N_CONV_POINTS"]

    def save_ckpt(store):
        tmp = ckpt + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(store, fh)
        os.replace(tmp, ckpt)

    if os.path.exists(ckpt):
        with open(ckpt) as fh:
            store = json.load(fh)
        bad = [k for k in keys if store["config"].get(k) != config[k]]
        if bad:
            raise RuntimeError(f"Le checkpoint existant a été produit avec d'autres paramètres {bad}. "
                               f"Changez le dossier de sortie ou supprimez {ckpt}.")
        store["config"] = config
        n_done = sum(len(v) for v in store["runs"].values())
        print(f"Reprise depuis le checkpoint : {n_done} runs déjà terminés.")
    else:
        store = dict(config=config, runs={}, started=datetime.datetime.now().isoformat(timespec="seconds"))

    t_start, last_save = time.time(), time.time()
    n_total, n_new = len(cfg.func_ids) * cfg.runs, 0
    for fid in cfg.func_ids:
        fobj = cfg.make(fid)
        done = store["runs"].setdefault(str(fid), {})
        for run in range(cfg.runs):
            if str(run) in done:
                continue
            rng = np.random.default_rng(np.random.SeedSequence([BASE_SEED, fid, run]))
            t0 = time.perf_counter()
            with np.errstate(all="ignore"):
                res = run_abcng(fobj, cfg.D, fobj.lb, fobj.ub, fobj.f_opt, SN, LIMIT, cfg.max_fes, rng)
            dt = time.perf_counter() - t0
            done[str(run)] = dict(best_err=res["best_err"], best_f=res["best_f"], fes=res["fes"],
                                  time_s=dt, final_k=res["final_k"],
                                  best_x=res["best_x"].tolist(), hist=res["hist"].tolist())
            n_new += 1
            if time.time() - last_save >= CHECKPOINT_EVERY_S:
                save_ckpt(store); last_save = time.time()
                print(f"   [checkpoint horaire enregistré {datetime.datetime.now():%H:%M:%S}]")
        save_ckpt(store); last_save = time.time()
        errs = np.array([done[str(r)]["best_err"] for r in range(cfg.runs)])
        tms  = np.array([done[str(r)]["time_s"] for r in range(cfg.runs)])
        n_fin = sum(len(v) for v in store["runs"].values())
        el = time.time() - t_start
        eta = el / max(n_new, 1) * (n_total - n_fin) if n_new else 0.0
        print(f"F{fid:>2} {cfg.types[fid][:5]:<5} | mean = {errs.mean():.3e} | std = {errs.std(ddof=STD_DDOF):.3e} "
              f"| best = {errs.min():.3e} | t/run = {tms.mean():6.1f} s | ETA ≈ {eta/60:6.1f} min")

    store["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
    save_ckpt(store)
    total_cpu_time = sum(r["time_s"] for f in store["runs"].values() for r in f.values())
    print(f"\nExpérience terminée. Temps cumulé des runs : {total_cpu_time/3600:.2f} h")
    return store, config


# %%
def analyse(cfg, store):
    rows = []
    for fid in cfg.func_ids:
        for run in range(cfg.runs):
            r = store["runs"][str(fid)][str(run)]
            rows.append(dict(Function=fid, Run=run + 1, Seed=f"[{BASE_SEED},{fid},{run}]",
                             Error=r["best_err"], BestF=r["best_f"], FEs=r["fes"], Time_s=r["time_s"]))
    df_runs = pd.DataFrame(rows)

    summ = []
    for fid in cfg.func_ids:
        e = df_runs.loc[df_runs.Function == fid, "Error"].to_numpy()
        t = df_runs.loc[df_runs.Function == fid, "Time_s"].to_numpy()
        summ.append(dict(Function=f"F{fid}", Name=cfg.names[fid], Type=cfg.types[fid],
                         Best=e.min(), Median=np.median(e), Worst=e.max(), Mean=e.mean(),
                         Std=e.std(ddof=STD_DDOF), MeanTime_s=t.mean(), StdTime_s=t.std(ddof=STD_DDOF),
                         TotalTime_s=t.sum()))
    df_sum = pd.DataFrame(summ)

    with pd.option_context("display.float_format", "{:.3e}".format, "display.width", 200):
        print(df_sum[["Function", "Type", "Best", "Median", "Worst", "Mean", "Std", "MeanTime_s"]].to_string(index=False))
    print(f"\nTemps total (somme des runs) : {df_runs.Time_s.sum()/3600:.2f} h | "
          f"temps moyen par run : {df_runs.Time_s.mean():.1f} s")

    H = {fid: np.array([store["runs"][str(fid)][str(r)]["hist"] for r in range(cfg.runs)]) for fid in cfg.func_ids}
    return df_runs, df_sum, H


# %%
FLOOR = 1e-16


def make_figures(cfg, df_runs, df_sum, H):
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.grid": True, "grid.alpha": 0.3,
                         "axes.titlesize": 8, "figure.dpi": 100})
    marks = conv_marks(cfg.max_fes)

    def plot_conv(ax, fid):
        h = np.maximum(H[fid], FLOOR)
        q1, q3 = np.percentile(h, [25, 75], axis=0)
        ax.plot(marks, h.mean(axis=0), lw=1.3, color="tab:blue", label="Mean")
        ax.plot(marks, np.median(h, axis=0), lw=1.0, ls="--", color="tab:red", label="Median")
        ax.fill_between(marks, q1, q3, alpha=0.2, color="tab:blue", label="IQR")
        ax.set_yscale("log"); ax.set_title(f"F{fid}: {cfg.names[fid]}", fontsize=7)
        ax.set_xlabel("FEs"); ax.set_ylabel("Error")

    for fid in cfg.func_ids:
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        plot_conv(ax, fid); ax.legend(fontsize=6)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(cfg.fig_dir, "convergence", f"F{fid:02d}.{ext}"), dpi=300)
        plt.close(fig)

    ncol = 4; nrow = math.ceil(len(cfg.func_ids) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 2.5 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, fid in zip(axes, cfg.func_ids):
        plot_conv(ax, fid)
    for ax in axes[len(cfg.func_ids):]: ax.axis("off")
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(cfg.fig_dir, f"convergence_all.{ext}"), dpi=300)
    plt.show()

    order = []
    for f in cfg.func_ids:
        if cfg.types[f] not in order: order.append(cfg.types[f])
    groups = [(t, [f for f in cfg.func_ids if cfg.types[f] == t]) for t in order]
    fig, axes = plt.subplots(1, len(groups), figsize=(14, 3.2),
                             gridspec_kw=dict(width_ratios=[max(len(g), 3) for _, g in groups]))
    axes = np.atleast_1d(axes)
    for ax, (name, g) in zip(axes, groups):
        data = [np.log10(np.maximum(df_runs.loc[df_runs.Function == f, "Error"].to_numpy(), FLOOR)) for f in g]
        ax.boxplot(data, showmeans=True, flierprops=dict(markersize=2))
        ax.set_xticks(range(1, len(g) + 1)); ax.set_xticklabels([f"F{f}" for f in g])
        ax.set_title(name); ax.set_ylabel(r"$\log_{10}$(error)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(cfg.fig_dir, f"boxplots.{ext}"), dpi=300)
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(df_sum.Function, df_sum.MeanTime_s, yerr=df_sum.StdTime_s, capsize=2, color="tab:gray")
    ax.set_ylabel("Time per run (s)")
    ax.set_title(f"{ALGO_NAME} — mean execution time per run ({cfg.bench}, D={cfg.D}, MaxFEs={cfg.max_fes})")
    plt.setp(ax.get_xticklabels(), rotation=60)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(cfg.fig_dir, f"execution_time.{ext}"), dpi=300)
    plt.show()


# %%
def tex_sci(v):
    m, e = f"{v:.2e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$" if v != 0 else "$0$"


def _cpu():
    try:
        with open("/proc/cpuinfo") as fh:
            for l in fh:
                if l.lower().startswith("model name"): return l.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def export_results(cfg, store, config, df_runs, df_sum, H):
    P = lambda name: os.path.join(cfg.out, name)
    marks = conv_marks(cfg.max_fes)
    tag = cfg.tag

    df_runs.to_csv(P(f"{tag}_all_runs.csv"), index=False)
    df_sum.to_csv(P(f"{tag}_summary.csv"), index=False)

    conv = pd.DataFrame({"FEs": marks})
    for fid in cfg.func_ids:
        conv[f"F{fid}_mean"] = H[fid].mean(axis=0)
    conv.to_csv(P(f"{tag}_convergence_mean.csv"), index=False)

    np.savez_compressed(P(f"{tag}_raw_arrays.npz"),
                        funcs=np.array(cfg.func_ids), marks=marks,
                        errors=np.array([[store["runs"][str(f)][str(r)]["best_err"] for r in range(cfg.runs)] for f in cfg.func_ids]),
                        times=np.array([[store["runs"][str(f)][str(r)]["time_s"] for r in range(cfg.runs)] for f in cfg.func_ids]),
                        histories=np.array([H[f] for f in cfg.func_ids]),
                        best_x=np.array([[store["runs"][str(f)][str(r)]["best_x"] for r in range(cfg.runs)] for f in cfg.func_ids]))

    lines = [r"\begin{table}[htbp]", r"\centering",
             rf"\caption{{Error values of {ALGO_NAME} on the {cfg.bench} benchmark ($D={cfg.D}$, SN$={SN}$, MaxFEs$={cfg.max_fes}$, "
             rf"limit$={LIMIT}$, {cfg.runs} independent runs).}}",
             rf"\label{{tab:{tag}}}", r"\small", r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"F & Type & Best & Median & Worst & Mean & Std \\", r"\midrule"]
    for _, r in df_sum.iterrows():
        lines.append(f"{r.Function} & {r.Type} & {tex_sci(r.Best)} & {tex_sci(r.Median)} & {tex_sci(r.Worst)} & "
                     f"{tex_sci(r.Mean)} & {tex_sci(r.Std)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(P(f"{tag}_table.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    env = dict(python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__,
               matplotlib=matplotlib.__version__, platform=platform.platform(), cpu=_cpu(),
               cpu_count=os.cpu_count(), date=datetime.datetime.now().isoformat(timespec="seconds"))
    with open(P(f"{tag}_config_environment.json"), "w") as fh:
        json.dump(dict(config=config, environment=env,
                       total_run_time_s=float(df_runs.Time_s.sum()),
                       notes=["Raw errors f(x)-f* (no 1e-8 truncation).",
                              "Sequential implementation (no parallelism).",
                              "Seeds: numpy SeedSequence([BASE_SEED, function, run])."]), fh, indent=2)
    print("Fichiers CSV / NPZ / JSON / LaTeX écrits dans", cfg.out)
    return env


# %%
try:
    from docx import Document
    from docx.shared import Pt, Inches
    from docx.enum.text import WD_ALIGN_PARAGRAPH
except ImportError:
    Document = None
    print("python-docx absent : pip install python-docx  (rapport .docx ignoré)")


def sci(v): return f"{v:.2e}"


def add_table(doc, header, body, font_pt=8):
    t = doc.add_table(rows=1, cols=len(header)); t.style = "Table Grid"
    for c, h in zip(t.rows[0].cells, header):
        c.text = ""; r = c.paragraphs[0].add_run(h); r.bold = True; r.font.size = Pt(font_pt)
    for row in body:
        cells = t.add_row().cells
        for c, v in zip(cells, row):
            c.text = ""; c.paragraphs[0].add_run(str(v)).font.size = Pt(font_pt)
            c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    return t


def make_report(cfg, df_runs, df_sum, env):
    if Document is None:
        return
    name = DISPLAY[cfg.bench]
    doc = Document()
    doc.styles["Normal"].font.name = "Times New Roman"; doc.styles["Normal"].font.size = Pt(10)
    doc.add_heading(f"{ALGO_NAME} on the {name} benchmark (D = {cfg.D})", level=1)

    doc.add_heading("Experimental setup", level=2)
    doc.add_paragraph(
        f"{ALGO_NAME} was evaluated on the {len(cfg.func_ids)} {name} single-objective bound-constrained functions "
        f"in {cfg.D} dimensions (search range [-100, 100]^{cfg.D}). Each function was solved in {cfg.runs} independent "
        f"runs with a population size SN = {SN}, limit = {LIMIT} and a termination criterion of MaxFEs = {cfg.max_fes} "
        "function evaluations. Reported errors are f(x) - f(x*) and are not truncated at 1e-8. All runs were "
        "executed sequentially (no parallelism), with run-specific random seeds.")
    add_table(doc, ["Parameter", "Value"],
              [["Dimension D", cfg.D], ["Population size SN", SN], ["MaxFEs", cfg.max_fes], ["limit", LIMIT],
               ["Independent runs", cfg.runs], ["Initial neighborhood radius k", K_INIT],
               ["psi range", f"[0, {PSI_MAX}]"], ["CPU", env["cpu"]], ["Python / NumPy", f"{env['python']} / {env['numpy']}"]])

    doc.add_heading("Results", level=2)
    doc.add_paragraph(f"Table 1. Error values (best, median, worst, mean, standard deviation) of {ALGO_NAME} over {cfg.runs} runs.")
    add_table(doc, ["F", "Type", "Best", "Median", "Worst", "Mean", "Std"],
              [[r.Function, r.Type, sci(r.Best), sci(r.Median), sci(r.Worst), sci(r.Mean), sci(r.Std)]
               for _, r in df_sum.iterrows()])

    doc.add_paragraph()
    doc.add_paragraph(f"Table 2. Execution time per run (seconds), D = {cfg.D}.")
    body = [[r.Function, f"{r.MeanTime_s:.2f}", f"{r.StdTime_s:.2f}", f"{r.TotalTime_s:.1f}"] for _, r in df_sum.iterrows()]
    body.append(["All", f"{df_runs.Time_s.mean():.2f}", f"{df_runs.Time_s.std(ddof=STD_DDOF):.2f}", f"{df_runs.Time_s.sum():.1f}"])
    add_table(doc, ["F", "Mean time (s)", "Std time (s)", "Total (s)"], body)

    for cap, fn in [("Figure 1. Convergence curves (mean, median and interquartile range of the error).", "convergence_all.png"),
                    ("Figure 2. Boxplots of the final errors (log10 scale).", "boxplots.png"),
                    ("Figure 3. Mean execution time per run.", "execution_time.png")]:
        doc.add_page_break()
        doc.add_picture(os.path.join(cfg.fig_dir, fn), width=Inches(6.3))
        doc.add_paragraph(cap)

    docx_path = os.path.join(cfg.out, f"{cfg.tag}_report.docx")
    doc.save(docx_path)
    print("Rapport Word :", docx_path)


# %%
def make_archive(cfg):
    zip_path = f"{cfg.out}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(cfg.out):
            for fn in files:
                if fn.endswith(".tmp"): continue
                fp = os.path.join(root, fn)
                z.write(fp, os.path.relpath(fp, os.path.dirname(cfg.out) or "."))
    print("Archive :", zip_path)
    for root, _, files in os.walk(cfg.out):
        if "convergence" in root: continue
        for fn in sorted(files):
            print(f"  {os.path.join(root, fn):<80} {os.path.getsize(os.path.join(root, fn))/1024:9.1f} Ko")


# %%
for bench, dim in EXPERIMENTS:
    print("\n" + "=" * 100)
    cfg = build_cfg(bench, dim)
    store, config = execute(cfg)
    df_runs, df_sum, H = analyse(cfg, store)
    make_figures(cfg, df_runs, df_sum, H)
    env = export_results(cfg, store, config, df_runs, df_sum, H)
    make_report(cfg, df_runs, df_sum, env)
    make_archive(cfg)
