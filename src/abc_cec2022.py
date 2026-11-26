import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
from cec2022_official import CEC2022

warnings.filterwarnings("ignore")

VARIANT_NAME = "ABC"
SN = 50
LIMIT = 100
OUTPUT_ROOT = "./abc_results"

BENCHMARKS = {
    "cec2022": {
        "cls": CEC2022,
        "funcs": list(range(1, 13)),
        "runs": 30,
        "dims": [10],
        "max_fes": {10: 200_000},
    },
}

TO_RUN = ["cec2022"]

RUN_COLS = ["variant", "benchmark", "D", "func_id", "run_idx", "seed",
            "final_error", "runtime_sec", "timestamp"]


def fmt_time(sec):
    sec = int(round(sec))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h}h {m:02d}min {s:02d}s"


class ABC:
    def __init__(self, problem, D, MaxFEs, SN=50, limit=100):
        self.fn = problem
        self.f_opt = float(problem.f_opt)
        self.lb = float(problem.lb[0])
        self.ub = float(problem.ub[0])
        self.D, self.SN, self.limit = D, SN, limit
        self.MaxFEs = MaxFEs

    def _eval(self, x):
        v = self.fn(x)
        return v if np.isfinite(v) else 1e300

    def run(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        SN, D, lb, ub = self.SN, self.D, self.lb, self.ub
        limit, MaxFEs, f_opt = self.limit, self.MaxFEs, self.f_opt
        ev = self._eval
        rand_int = np.random.randint
        rand_sample = np.random.random_sample
        rand_uniform = np.random.uniform
        sn1 = SN - 1

        pop = rand_uniform(lb, ub, (SN, D))
        fvals = [ev(pop[i]) for i in range(SN)]
        trial = [0] * SN
        NFE = SN
        best_val = min(fvals)
        history = [max(0.0, best_val - f_opt)]

        while NFE < MaxFEs:
            for i in range(SN):
                if NFE >= MaxFEs:
                    break
                k = rand_int(sn1)
                if k >= i:
                    k += 1
                j = rand_int(D)
                r = -1.0 + 2.0 * rand_sample()
                xi = pop[i]
                xij = xi[j]
                nv = xij + r * (xij - pop[k, j])
                if nv < lb:
                    nv = lb
                elif nv > ub:
                    nv = ub
                xi[j] = nv
                fv = ev(xi)
                NFE += 1
                if fv <= fvals[i]:
                    fvals[i] = fv
                    trial[i] = 0
                    if fv < best_val:
                        best_val = fv
                else:
                    xi[j] = xij
                    trial[i] += 1

            fa = np.array(fvals)
            fit = np.where(fa >= 0, 1.0 / (1.0 + fa), 1.0 + np.abs(fa))
            s = np.sum(fit)
            if not np.isfinite(s) or s < 1e-12:
                pr = np.full(SN, 1.0 / SN)
            else:
                pr = np.clip(fit, 0, None)
                pr = pr / (pr.sum() + 1e-12)
                pr = pr / pr.sum()
            cdf = pr.cumsum()
            cdf /= cdf[-1]
            search = cdf.searchsorted
            for _ in range(SN):
                if NFE >= MaxFEs:
                    break
                i = int(search(rand_sample(), side="right"))
                i = min(i, sn1)
                k = rand_int(sn1)
                if k >= i:
                    k += 1
                j = rand_int(D)
                r = -1.0 + 2.0 * rand_sample()
                xi = pop[i]
                xij = xi[j]
                nv = xij + r * (xij - pop[k, j])
                if nv < lb:
                    nv = lb
                elif nv > ub:
                    nv = ub
                xi[j] = nv
                fv = ev(xi)
                NFE += 1
                if fv <= fvals[i]:
                    fvals[i] = fv
                    trial[i] = 0
                    if fv < best_val:
                        best_val = fv
                else:
                    xi[j] = xij
                    trial[i] += 1

            for i in range(SN):
                if NFE >= MaxFEs:
                    break
                if trial[i] > limit:
                    pop[i] = rand_uniform(lb, ub, D)
                    fv = ev(pop[i])
                    fvals[i] = fv
                    NFE += 1
                    trial[i] = 0
                    if fv < best_val:
                        best_val = fv

            history.append(max(0.0, best_val - f_opt))

        err = max(0.0, best_val - f_opt)
        return err, np.array(history)


def out_dir(bench, D):
    d = os.path.join(OUTPUT_ROOT, f"{bench}_D{D}")
    os.makedirs(d, exist_ok=True)
    return d


def runs_path(bench, D):
    return os.path.join(out_dir(bench, D), f"abc_runs_{bench}_D{D}.csv")


def summary_path(bench, D):
    return os.path.join(out_dir(bench, D), f"abc_{bench}_D{D}.csv")


def aggregate(bench, D):
    p = runs_path(bench, D)
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    if df.empty:
        return None
    g = df.groupby("func_id")
    summary = pd.DataFrame({
        "mean": g["final_error"].mean(),
        "std": g["final_error"].std().fillna(0.0),
        "best": g["final_error"].min(),
        "worst": g["final_error"].max(),
        "median": g["final_error"].median(),
        "n_runs": g["final_error"].count(),
        "mean_time_sec": g["runtime_sec"].mean(),
        "total_time_sec": g["runtime_sec"].sum(),
    }).reset_index()
    summary.insert(0, "D", D)
    summary.insert(0, "benchmark", bench)
    summary.insert(0, "variant", VARIANT_NAME)
    summary.to_csv(summary_path(bench, D), index=False)
    return summary


def quick_test():
    for bench in TO_RUN:
        cfg = BENCHMARKS[bench]
        D = min(cfg["dims"] + [10])
        for fid in cfg["funcs"][:2]:
            prob = cfg["cls"](func_num=fid, dim=D)
            t0 = time.perf_counter()
            err, hist = ABC(prob, D, MaxFEs=2000 * D, SN=SN, limit=LIMIT).run(seed=1)
            assert np.isfinite(err) and len(hist) > 1
            print(f"[{bench}] F{fid:<2d} D={D}  error={err:.4e}  time={time.perf_counter() - t0:.2f}s")
    print("Test OK")


def run_benchmark(bench):
    cfg = BENCHMARKS[bench]
    for D in cfg["dims"]:
        p = runs_path(bench, D)
        done = set()
        if os.path.exists(p):
            old = pd.read_csv(p)
            done = set(zip(old.func_id, old.run_idx))
        else:
            pd.DataFrame(columns=RUN_COLS).to_csv(p, index=False)

        for func_id in cfg["funcs"]:
            problem = cfg["cls"](func_num=func_id, dim=D)
            for run_idx in range(cfg["runs"]):
                if (func_id, run_idx) in done:
                    continue
                seed = 100000 * func_id + 1000 * D + run_idx
                t0 = time.perf_counter()
                err, _ = ABC(problem, D, MaxFEs=cfg["max_fes"][D],
                             SN=SN, limit=LIMIT).run(seed=seed)
                rt = time.perf_counter() - t0

                row = pd.DataFrame([{
                    "variant": VARIANT_NAME, "benchmark": bench, "D": D,
                    "func_id": func_id, "run_idx": run_idx, "seed": seed,
                    "final_error": err, "runtime_sec": rt,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }])
                row.to_csv(p, mode="a", header=False, index=False)
                print(f"[{bench}] D={D:<2d} F{func_id:<2d} run {run_idx + 1:>2d}/{cfg['runs']}"
                      f"  error={err:.4e}  time={rt:.1f}s", flush=True)

            aggregate(bench, D)


def print_summaries(summaries):
    for (bench, D), summary in summaries.items():
        if summary is None:
            print(f"{bench} D={D}: no results.")
            continue
        print(f"\n=== {VARIANT_NAME} -- {bench.upper()}, D={D} ===")
        show = summary.copy()
        show["mean_time"] = show["mean_time_sec"].map(lambda v: f"{v:.1f}s")
        show["total_time"] = show["total_time_sec"].map(fmt_time)
        print(show[["func_id", "mean", "std", "best", "worst", "median",
                    "n_runs", "mean_time", "total_time"]].to_string(index=False))
        print(f"Total compute time: {fmt_time(summary['total_time_sec'].sum())}")
        print("Mean")
        for v in summary["mean"]:
            if v == 0:
                print("0.00×10⁰")
            else:
                exponent = int(np.floor(np.log10(abs(v))))
                mantissa = v / (10 ** exponent)
                print(f"{mantissa:.2f}×10^{exponent}")


def main():
    quick_test()
    t_start = time.perf_counter()
    try:
        for bench in TO_RUN:
            run_benchmark(bench)
        print(f"\nElapsed (this session): {fmt_time(time.perf_counter() - t_start)}")
        print("Done. Results in", os.path.abspath(OUTPUT_ROOT))
    except KeyboardInterrupt:
        print("\nInterrupted -- progress is saved, rerun to resume.")
    finally:
        summaries = {(b, D): aggregate(b, D)
                     for b in TO_RUN for D in BENCHMARKS[b]["dims"]}
        print_summaries(summaries)


if __name__ == "__main__":
    main()
