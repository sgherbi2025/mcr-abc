import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

VARIANT_NAME = "ABC"
SUMMARY_FILES = {"cec2022": "abc_cec_2022_D10.csv"}
SN = 50
LIMIT = 100
RUNS = 30
BASE_SEED = 2022

SUITES = {
    "cec2022": {"D": 10, "funcs": range(1, 13), "max_fes": 200_000},
}


def make_problem(suite, func_id, D):
    from cec2022_official import CEC2022
    return CEC2022(func_num=func_id, dim=D)


class Optimizer:
    def __init__(self, problem, D, MaxFEs, SN=SN, limit=LIMIT):
        self.fn = problem
        self.f_opt = float(problem.f_opt)
        self.lb = float(np.ravel(problem.lb)[0])
        self.ub = float(np.ravel(problem.ub)[0])
        self.D, self.SN, self.limit, self.MaxFEs = D, SN, limit, MaxFEs

    def evaluate(self, x):
        if self.NFE >= self.MaxFEs:
            raise RuntimeError("budget depasse")
        v = float(self.fn(x.copy()))
        self.NFE += 1
        v = v if np.isfinite(v) else 1e300
        if v < self.best_val:
            self.best_val = v
        return v

    def other(self, i):
        k = self.rng.integers(self.SN - 1)
        return k + 1 if k >= i else k

    def search(self, i):
        j = self.rng.integers(self.D)
        k = self.other(i)
        phi = -1.0 + 2.0 * self.rng.random()
        xi = self.pop[i]
        old = xi[j]
        new = old + phi * (old - self.pop[k, j])
        xi[j] = min(max(new, self.lb), self.ub)
        fv = self.evaluate(xi)
        if fv <= self.fvals[i]:
            self.fvals[i] = fv
            self.trial[i] = 0
        else:
            xi[j] = old
            self.trial[i] += 1

    def restart(self, i, x):
        self.pop[i] = x
        self.fvals[i] = self.evaluate(self.pop[i])
        self.trial[i] = 0
        self.restarts += 1

    def scout(self):
        i = int(np.argmax(self.trial))
        if self.trial[i] > self.limit and self.NFE < self.MaxFEs:
            self.restart(i, self.lb + self.rng.random(self.D) * (self.ub - self.lb))

    def run(self, seed):
        self.rng = np.random.default_rng(np.random.SeedSequence(list(seed)))
        SN, D = self.SN, self.D
        self.NFE, self.restarts, self.best_val = 0, 0, np.inf
        self.pop = self.lb + self.rng.random((SN, D)) * (self.ub - self.lb)
        self.fvals = [self.evaluate(self.pop[i]) for i in range(SN)]
        self.trial = [0] * SN

        while self.NFE < self.MaxFEs:
            for i in range(SN):
                if self.NFE >= self.MaxFEs:
                    break
                self.search(i)

            fa = np.array(self.fvals)
            fit = np.where(fa >= 0, 1.0 / (1.0 + np.abs(fa)), 1.0 + np.abs(fa))
            cum = fit.cumsum()
            for _ in range(SN):
                if self.NFE >= self.MaxFEs:
                    break
                i = int(cum.searchsorted(self.rng.random() * cum[-1], side="right"))
                self.search(min(i, SN - 1))

            self.scout()

        return {"final_error": self.best_val - self.f_opt,
                "restarts": self.restarts, "nfe": self.NFE}


def run_one(suite, func_id, run_idx, max_fes):
    t0 = time.perf_counter()
    D = SUITES[suite]["D"]
    problem = make_problem(suite, func_id, D)
    seed = (BASE_SEED, func_id, run_idx)
    res = Optimizer(problem, D, max_fes).run(seed)
    return {"variant": VARIANT_NAME, "suite": suite, "D": D, "func_id": func_id,
            "run_idx": run_idx, "seed": "-".join(map(str, seed)), **res,
            "max_fes": max_fes, "runtime_sec": time.perf_counter() - t0,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}


def summarise(runs_file, summary_file, suite):
    g = pd.read_csv(runs_file).groupby("func_id")
    e, t = g.final_error, g.runtime_sec
    s = pd.DataFrame({
        "mean": e.mean(), "std": e.std(), "best": e.min(), "worst": e.max(),
        "median": e.median(), "n_runs": e.count(),
        "mean_time_sec": t.mean(), "total_time_sec": t.sum(),
    }).reset_index()
    s.insert(0, "D", SUITES[suite]["D"])
    s.insert(0, "variant", VARIANT_NAME)
    s.to_csv(summary_file, index=False)
    print(s.to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=VARIANT_NAME)
    ap.add_argument("--suite", choices=["cec2022"], default="cec2022")
    ap.add_argument("--funcs", type=int, nargs="+")
    ap.add_argument("--runs", type=int, default=RUNS)
    ap.add_argument("--max-fes", type=int)
    ap.add_argument("--output", default="./Results")
    a = ap.parse_args()
    os.makedirs(a.output, exist_ok=True)

    for suite in [a.suite]:
        cfg = SUITES[suite]
        max_fes = a.max_fes or cfg["max_fes"]
        summary_file = os.path.join(a.output, SUMMARY_FILES[suite])
        runs_file = summary_file[:-4] + "_runs.csv"
        done = set()
        if os.path.exists(runs_file):
            old = pd.read_csv(runs_file)
            if (old.max_fes != max_fes).any():
                raise SystemExit(f"{runs_file} : autre budget, changez --output")
            done = set(zip(old.func_id, old.run_idx))
        todo = [(f, r) for f in (a.funcs or cfg["funcs"])
                for r in range(a.runs) if (f, r) not in done]
        print(f"{VARIANT_NAME} {suite} : {len(todo)} runs a faire, {len(done)} deja faits")

        for n, (f, r) in enumerate(todo, 1):
            row = run_one(suite, f, r, max_fes)
            pd.DataFrame([row]).to_csv(runs_file, mode="a", index=False,
                                       header=not os.path.exists(runs_file))
            print(f"[{n}/{len(todo)}] F{f} run {r} erreur={row['final_error']:.6e} "
                  f"redemarrages={row['restarts']}", flush=True)
        if os.path.exists(runs_file):
            summarise(runs_file, summary_file, suite)


if __name__ == "__main__":
    main()
