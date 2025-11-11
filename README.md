\# PyNetIM



A Python library for Network Influence Maximization (IM) problems, supporting multiple diffusion models (IC, LT) and algorithms (heuristic, RIS, simulation).



\## Installation



```bash

pip install PyNetIM

# Quick Start

from PyNetIM.diffusion\\\_model import IndependentCascadeModel

from PyNetIM.graph import Graph


g = Graph()

# ... build graph

model = IndependentCascadeModel(g, p=0.1)

seeds = model.greedy(k=10)


# Features

- Independent Cascade (IC)

* Linear Threshold (LT)
* Monte Carlo simulation
* RIS algorithm
* Heuristic methods



