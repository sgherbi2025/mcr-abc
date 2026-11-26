import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from cec2013_functions import CEC2013
from cec2022_official import CEC2022

RNG_MASTER_SEED = 42
SN = 50
LIMIT = 100
NEIGH_N = 5
N_RUNS = 30
MAX_NFE_CEC2013_FACTOR = 10000
MAX_NFE_CEC2022 = {10: 200000}

def fitness_transform(f):
    return np.where(f >= 0, 1.0 / (1.0 + f), 1.0 + np.abs(f))

def nabc(evaluate, lb, ub, D, SN, limit, N, max_nfe, rng):
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    X = lb + rng.random((SN, D)) * (ub - lb)
    fit = np.empty(SN)
    for i in range(SN):
        fit[i] = evaluate(X[i])
    nfe = SN
    trial = np.zeros(SN, dtype=int)
    all_idx = np.arange(SN)

    def best_neighbor_move(i, X, fit):
        candidates = rng.choice(SN - 1, size=min(N, SN - 1), replace=False)
        candidates = np.where(candidates >= i, candidates + 1, candidates)
        nbest = candidates[np.argmin(fit[candidates])]
        j = rng.integers(0, D)
        phi = rng.uniform(-1.0, 1.0)
        V = X[i].copy()
        V[j] = X[nbest, j] + phi * (X[nbest, j] - X[i, j])
        np.clip(V, lb, ub, out=V)
        return V

    while nfe < max_nfe:
        for i in range(SN):
            if nfe >= max_nfe:
                break
            V = best_neighbor_move(i, X, fit)
            fV = evaluate(V)
            nfe += 1
            if fit[i] > fV:
                X[i] = V
                fit[i] = fV
                trial[i] = 0
            else:
                trial[i] += 1
        if nfe >= max_nfe:
            break
        fitv = fitness_transform(fit)
        p = fitv / fitv.sum()
        count = 0
        while count < SN and nfe < max_nfe:
            j = rng.choice(SN, p=p)
            V = best_neighbor_move(j, X, fit)
            fV = evaluate(V)
            nfe += 1
            if fit[j] > fV:
                X[j] = V
                fit[j] = fV
                trial[j] = 0
            else:
                trial[j] += 1
            count += 1
        if nfe >= max_nfe:
            break
        gbest_idx = np.argmin(fit)
        for i in range(SN):
            if nfe >= max_nfe:
                break
            if trial[i] > limit:
                trial[i] = 0
                others = all_idx[all_idx != i]
                a, b = rng.choice(others, size=2, replace=False)
                r = rng.random(3)
                r = r / r.sum()
                V = r[0] * X[i] + r[1] * X[gbest_idx] + r[2] * (X[a] - X[b])
                np.clip(V, lb, ub, out=V)
                fV = evaluate(V)
                nfe += 1
                X[i] = V
                fit[i] = fV
    return float(np.min(fit))

def build_function(suite, func_num, D):
    if suite == "cec2013":
        return CEC2013(func_num=func_num, dim=D)
    return CEC2022(func_num=func_num, dim=D)

def max_nfe_for(suite, D):
    if suite == "cec2013":
        return MAX_NFE_CEC2013_FACTOR * D
    return MAX_NFE_CEC2022[D]

def run_single(f_inst, suite, func_num, run_idx, D):
    seed = RNG_MASTER_SEED * 10000 + func_num * 100 + run_idx
    rng = np.random.default_rng(seed)
    evaluate = lambda x: float(f_inst(np.asarray(x, dtype=float)))
    best = nabc(evaluate, f_inst.lb, f_inst.ub, D, SN, LIMIT, NEIGH_N, max_nfe_for(suite, D), rng)
    return max(best - f_inst.f_opt, 0.0)

def run_suite(suite, D, n_funcs, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in tqdm(range(1, n_funcs + 1), desc=f"{suite.upper()} D={D}"):
        f_inst = build_function(suite, i, D)
        errors = np.array([run_single(f_inst, suite, i, r, D) for r in range(N_RUNS)])
        rows.append({
            "Function": f"f{i}",
            "Name": getattr(f_inst, "name", f"F{i}"),
            "D": D,
            "Runs": len(errors),
            "Mean Error": errors.mean(),
            "Std Dev": errors.std(ddof=1) if len(errors) > 1 else 0.0,
            "Best": errors.min(),
            "Worst": errors.max(),
        })
    df = pd.DataFrame(rows)
    for col in ["Mean Error", "Std Dev", "Best", "Worst"]:
        df[col] = df[col].map(lambda x: f"{x:.6E}")
    csv_path = output_dir / f"nabc_{suite}_D{D}.csv"
    df.to_csv(csv_path, index=False)
    print(f"CSV enregistre : {csv_path.resolve()}")

if __name__ == "__main__":
    run_suite("cec2013", 30, 28, "results")
    run_suite("cec2022", 10, 12, "results")
