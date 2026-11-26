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

VARIANT_NAME = "ABC-M2"
SUITE = "cec2022"
D = 10
SN = 50
LIMIT = 100
BASE_SEED = 2022                      # identique a MCR-ABC
FUNCS = list(range(1, 13))
RUNS = 30
MAX_FES = 200_000
RUN_COLS = ["variant", "suite", "D", "func_id", "run_idx", "seed",
            "final_error", "runtime_sec", "timestamp"]


def fmt_time(sec):
    sec = int(round(sec))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h}h {m:02d}min {s:02d}s"


def make_problem(func_id, dim):
    from cec2022_official import CEC2022
    return CEC2022(func_num=func_id, dim=dim)


class ABC_M2:
    """ABC canonique + restart de TOUTES les sources avec trial > limit
    a chaque cycle, par re-echantillonnage uniforme (Eq. 3).
    Meme ordre de tirages aleatoires que MCR_ABC (init, employees, spectatrices),
    donc meme population initiale pour un meme seed (BASE_SEED, func_id, run_idx)."""

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
            # ---- abeilles employees (Eq. 1)
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

            # ---- abeilles spectatrices (Eq. 2)
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

            # ---- abeilles eclaireuses : M2 = toutes les sources trial > limit,
            # restart uniforme Eq. (3) (x_best, x_r non utilises)
            for i in range(SN):
                if NFE >= MaxFEs:
                    break
                if trial[i] > limit:
                    pop[i] = lb + rand_sample(D) * (ub - lb)
                    fv = ev(pop[i])
                    fvals[i] = fv
                    NFE += 1
                    trial[i] = 0
                    if fv < best_val:
                        best_val = fv

            history.append(max(0.0, best_val - f_opt))

        err = max(0.0, best_val - f_opt)
        return err, np.array(history)


def runs_path(output_dir):
    return os.path.join(output_dir, f"abc_m2_runs_{SUITE}_D{D}.csv")


def summary_path(output_dir):
    return os.path.join(output_dir, f"abc_m2_{SUITE}_D{D}.csv")


def aggregate(output_dir):
    p = runs_path(output_dir)
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
    summary.insert(0, "suite", SUITE)
    summary.insert(0, "variant", VARIANT_NAME)
    summary.to_csv(summary_path(output_dir), index=False)
    return summary


def run_suite(funcs, runs, output_dir, max_fes):
    os.makedirs(output_dir, exist_ok=True)
    t_start = time.perf_counter()
    p = runs_path(output_dir)
    done = set()
    if os.path.exists(p):
        old = pd.read_csv(p)
        done = set(zip(old.func_id, old.run_idx))
    else:
        pd.DataFrame(columns=RUN_COLS).to_csv(p, index=False)

    for func_id in funcs:
        problem = make_problem(func_id, D)
        for run_idx in range(runs):
            if (func_id, run_idx) in done:
                continue
            seed = (BASE_SEED, func_id, run_idx)     # identique a MCR-ABC
            t0 = time.perf_counter()
            err, _ = ABC_M2(problem, D, SN=SN, MaxFEs=max_fes,
                            limit=LIMIT).run(seed=seed)
            rt = time.perf_counter() - t0
            row = pd.DataFrame([{
                "variant": VARIANT_NAME, "suite": SUITE, "D": D,
                "func_id": func_id, "run_idx": run_idx,
                "seed": "-".join(map(str, seed)),
                "final_error": err, "runtime_sec": rt,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }])
            row.to_csv(p, mode="a", header=False, index=False)
            print(f"{SUITE} D={D:<2d} F{func_id:<2d} run {run_idx + 1:>2d}/{runs}"
                  f"  error={err:.4e}  time={rt:.1f}s", flush=True)
        aggregate(output_dir)
    print(f"Elapsed: {fmt_time(time.perf_counter() - t_start)}")


def print_summary(output_dir):
    summary = aggregate(output_dir)
    if summary is None:
        return
    print(f"=== {VARIANT_NAME} -- {SUITE.upper()}, D={D} ===")
    show = summary.copy()
    show["mean_time"] = show["mean_time_sec"].map(lambda v: f"{v:.1f}s")
    show["total_time"] = show["total_time_sec"].map(fmt_time)
    print(show[["func_id", "mean", "std", "best", "worst", "median",
                "n_runs", "mean_time", "total_time"]].to_string(index=False))
    print(f"Total compute time D={D}: {fmt_time(summary['total_time_sec'].sum())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--funcs", type=int, nargs="+", default=None)
    ap.add_argument("--runs", type=int, default=None)
    ap.add_argument("--max-fes", type=int, default=None)
    ap.add_argument("--output", default="./abc_m2_results")
    args = ap.parse_args()

    funcs = args.funcs if args.funcs else FUNCS
    runs = args.runs if args.runs else RUNS
    max_fes = args.max_fes if args.max_fes else MAX_FES
    try:
        run_suite(funcs, runs, args.output, max_fes)
    except KeyboardInterrupt:
        print("Interrupted")
    print_summary(args.output)


if __name__ == "__main__":
    main()
