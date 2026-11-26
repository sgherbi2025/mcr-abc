
# MCR-ABC: Source Code and Experimental Results

This repository contains the Python source code and raw experimental results associated with the paper entitled "MCR-ABC: Memory-Driven Convex Restart for Artificial Bee Colony Optimization."

MCR-ABC is a variant of the Artificial Bee Colony (ABC) algorithm. Its performance is evaluated against standard ABC and its ablation/component variants (ABC-M1 and ABC-M2) on the CEC2022 benchmark suite ((D = 10)), as well as against several state-of-the-art competitors (ABCNG, MEABC, and NABC) on the CEC2013 ((D = 30)) and CEC2022 ((D = 10)) benchmark suites.

---

## Repository Structure

```text
mcr-abc/
├── src/
│   ├── mcr_abc_cec2013_cec2022.py   # Proposed MCR-ABC algorithm (CEC2013 & CEC2022)
│   ├── abc_cec2013_cec2022.py       # Standard ABC baseline
│   ├── abc_m1_core.py               # ABC-M1 variant (CEC2022)
│   ├── abc_m2_core.py               # ABC-M2 variant (CEC2022)
│   ├── abcng_cec2022_cec2013.py     # ABCNG competitor
│   ├── MEABC_CEC2013_2022.py        # MEABC competitor
│   └── nabc_cec2013_cec2022.py      # NABC competitor
├── Results/
│   ├── mcr-abc_cec2013_D30.csv      # MCR-ABC results on CEC2013 (D=30)
│   ├── mcr-abc_cec2022_D10.csv      # MCR-ABC results on CEC2022 (D=10)
│   ├── abc_cec2013_D30.csv          # ABC results on CEC2013 (D=30)
│   ├── abc_cec_2022_D10.csv         # ABC results on CEC2022 (D=10)
│   ├── abc-M1_cec2022_D10.csv       # ABC-M1 results on CEC2022 (D=10)
│   ├── abc-M2_cec2022_D10.csv       # ABC-M2 results on CEC2022 (D=10)
│   ├── abcng_cec2013_D30.csv        # ABCNG results on CEC2013 (D=30)
│   ├── abcng_cec2022_D10.csv        # ABCNG results on CEC2022 (D=10)
│   ├── meabc_cec2013_D30.csv        # MEABC results on CEC2013 (D=30)
│   ├── meabc_cec2022_D10.csv        # MEABC results on CEC2022 (D=10)
│   ├── nabc_cec2013_D30.csv         # NABC results on CEC2013 (D=30)
│   └── nabc_cec2022_D10.csv         # NABC results on CEC2022 (D=10)
└── README.txt

