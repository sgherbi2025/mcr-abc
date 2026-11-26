import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

warnings.filterwarnings("ignore")

VARIANT_NAME = "MCR-ABC"
SN = 50
LIMIT = 100
BASE_SEED = 2022
RUN_COLS = ["variant", "suite", "D", "func_id", "run_idx", "seed",
            "final_error", "runtime_sec", "timestamp"]

SUITES = {
    "cec2013": {
        "dims": [30],
        "funcs": list(range(1, 29)),
        "runs": 30,
        "max_fes": lambda D: 10000 * D,
    },
    "cec2022": {
        "dims": [10],
        "funcs": list(range(1, 13)),
        "runs": 30,
        "max_fes": lambda D: {10: 200_000}[D],
    },
}


def fmt_time(sec):
    sec = int(round(sec))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h}h {m:02d}min {s:02d}s"


def make_problem(suite, func_id, D):
    if suite == "cec2013":
        from cec2013_functions import CEC2013
        return CEC2013(func_num=func_id, dim=D)
    from cec2022_official import CEC2022
    return CEC2022(func_num=func_id, dim=D)


class MCR_ABC:
    def __init__(self, problem, D, SN=50, MaxFEs=None, limit=100):
        self.fn = problem
        self.f_opt = float(problem.f_opt)
        self.lb = float(problem.lb[0])
        self.ub = float(problem.ub[0])
        self.D, self.SN, self.limit = D, SN, limit
        self.MaxFEs = MaxFEs if MaxFEs is not None else 10000 * D

    def _eval(self, x):
        v = float(self.fn(x))
        return v if np.isfinite(v) else 1e300

    def run(self, seed=None):
        rng = np.random.default_rng(np.random.SeedSequence(list(seed)))
        SN, D, lb, ub = self.SN, self.D, self.lb, self.ub
        limit, MaxFEs, f_opt = self.limit, self.MaxFEs, self.f_opt
        ev = self._eval
        rand_int = rng.integers
        rand_sample = rng.random
        sn1 = SN - 1

        pop = lb + rng.random((SN, D)) * (ub - lb)
        fvals = [ev(pop[i]) for i in range(SN)]
        trial = [0] * SN
        NFE = SN
        best_val = min(fvals)
        history = [max(0.0, best_val - f_opt)]

        while NFE < MaxFEs:
            for i in range(SN):
                if NFE >= MaxFEs:
                    break
                j = rand_int(D)
                k = rand_int(sn1)
                if k >= i:
                    k += 1
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
            cum = fit.cumsum()
            tot = cum[-1]
            for _ in range(SN):
                if NFE >= MaxFEs:
                    break
                i = int(cum.searchsorted(rand_sample() * tot, side="right"))
                i = min(i, sn1)
                j = rand_int(D)
                k = rand_int(sn1)
                if k >= i:
                    k += 1
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

            # Scout phase: memory driven convex restart (Eq. 4)
            # x_best is fixed once, at the start of the scout phase
            x_best = pop[fvals.index(min(fvals))].copy()
            for i in range(SN):
                if NFE >= MaxFEs:
                    break
                if trial[i] > limit:
                    r_idx = rand_int(sn1)             # r ~ U{1..SN} \ {i}
                    if r_idx >= i:
                        r_idx += 1
                    x_r = pop[r_idx]
                    w = rand_sample(3)
                    x_rand = lb + rand_sample(D) * (ub - lb)
                    pop[i] = (w[0] * x_best + w[1] * x_r + w[2] * x_rand) / w.sum()
                    fv = ev(pop[i])
                    fvals[i] = fv
                    NFE += 1
                    trial[i] = 0
                    if fv < best_val:
                        best_val = fv

            history.append(max(0.0, best_val - f_opt))

        err = max(0.0, best_val - f_opt)
        return err, np.array(history)


def runs_path(output_dir, suite, D):
    return os.path.join(output_dir, f"mcr_abc_runs_{suite}_D{D}.csv")


def summary_path(output_dir, suite, D):
    return os.path.join(output_dir, f"mcr_abc_{suite}_D{D}.csv")


def aggregate(output_dir, suite, D):
    p = runs_path(output_dir, suite, D)
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
    summary.insert(0, "suite", suite)
    summary.insert(0, "variant", VARIANT_NAME)
    summary.to_csv(summary_path(output_dir, suite, D), index=False)
    return summary


def run_suite(suite, dims, funcs, runs, output_dir, max_fes_override=None):
    cfg = SUITES[suite]
    os.makedirs(output_dir, exist_ok=True)
    t_start = time.perf_counter()
    for D in dims:
        max_fes = max_fes_override if max_fes_override else cfg["max_fes"](D)
        p = runs_path(output_dir, suite, D)
        done = set()
        if os.path.exists(p):
            old = pd.read_csv(p)
            done = set(zip(old.func_id, old.run_idx))
        else:
            pd.DataFrame(columns=RUN_COLS).to_csv(p, index=False)

        for func_id in funcs:
            problem = make_problem(suite, func_id, D)
            for run_idx in range(runs):
                if (func_id, run_idx) in done:
                    continue
                seed = (BASE_SEED, func_id, run_idx)
                t0 = time.perf_counter()
                err, _ = MCR_ABC(problem, D, SN=SN, MaxFEs=max_fes,
                                 limit=LIMIT).run(seed=seed)
                rt = time.perf_counter() - t0
                row = pd.DataFrame([{
                    "variant": VARIANT_NAME, "suite": suite, "D": D,
                    "func_id": func_id, "run_idx": run_idx,
                    "seed": "-".join(map(str, seed)),
                    "final_error": err, "runtime_sec": rt,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }])
                row.to_csv(p, mode="a", header=False, index=False)
                print(f"{suite} D={D:<2d} F{func_id:<2d} run {run_idx + 1:>2d}/{runs}"
                      f"  error={err:.4e}  time={rt:.1f}s", flush=True)
            aggregate(output_dir, suite, D)
    print(f"Elapsed: {fmt_time(time.perf_counter() - t_start)}")


def print_summary(output_dir, suite, D):
    summary = aggregate(output_dir, suite, D)
    if summary is None:
        return
    print(f"=== {VARIANT_NAME} -- {suite.upper()}, D={D} ===")
    show = summary.copy()
    show["mean_time"] = show["mean_time_sec"].map(lambda v: f"{v:.1f}s")
    show["total_time"] = show["total_time_sec"].map(fmt_time)
    print(show[["func_id", "mean", "std", "best", "worst", "median",
                "n_runs", "mean_time", "total_time"]].to_string(index=False))
    print(f"Total compute time D={D}: {fmt_time(summary['total_time_sec'].sum())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["cec2013", "cec2022", "both"], default="both")
    ap.add_argument("--dims", type=int, nargs="+", default=None)
    ap.add_argument("--funcs", type=int, nargs="+", default=None)
    ap.add_argument("--runs", type=int, default=None)
    ap.add_argument("--max-fes", type=int, default=None)
    ap.add_argument("--output", default="./mcr_abc_results")
    args = ap.parse_args()

    suites = ["cec2013", "cec2022"] if args.suite == "both" else [args.suite]
    try:
        for suite in suites:
            cfg = SUITES[suite]
            dims = args.dims if args.dims else cfg["dims"]
            funcs = args.funcs if args.funcs else cfg["funcs"]
            runs = args.runs if args.runs else cfg["runs"]
            run_suite(suite, dims, funcs, runs, args.output, args.max_fes)
    except KeyboardInterrupt:
        print("Interrupted")
    for suite in suites:
        dims = args.dims if args.dims else SUITES[suite]["dims"]
        for D in dims:
            print_summary(args.output, suite, D)


if __name__ == "__main__":
    main()
