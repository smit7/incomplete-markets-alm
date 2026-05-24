# Mathematical Modelling of Incomplete Markets

This repository contains a quantitative finance simulation that prices a non-replicable digital option exhibiting basis risk. 

In incomplete markets, perfect risk replication is impossible. This project implements a single-period Asset-Liability Management (ALM) framework based on convex analysis to find the optimal stochastic hedge, minimizing the agent's expected shortfall penalty.

### Key Features:
* **Stochastic Approximation:** Utilizes the Robbins-Monro algorithm with diminishing step sizes to solve the underlying expected-value optimization where traditional calculus fails.
* **Realistic Market Frictions:** Models illiquidity via convex cost functions and handles strictly unhedgeable basis risk.
* **The Price Sandwich:** Computes and mathematically verifies the arbitrage-free pricing bounds: `Sub-hedging ≤ Accounting Value ≤ Indifference Price ≤ Super-hedging`.
* **Entropic Benchmark:** The computational outputs are successfully benchmarked against the closed-form entropic risk measure.

### How to Run:
The simulation is self-contained in `alm_pricing_model.py`. 
Requires: `numpy`
