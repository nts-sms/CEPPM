# CEPPM: Coupled Evolutionary Predator-Prey Model for Kiwi Conservation
Repository for Sierra Sharma's CEPPM EGT research. This repository contains the source code for the research paper: 
**"Behavioural Strategy and Predator Control: A Coupled Evolutionary Predator-Prey Model for Kiwi Conservation in New Zealand"**

## Project Overview
This model integrates **Replicator Dynamics** from evolutionary game theory with a **Lotka-Volterra** framework to simulate the co-evolution of kiwi foraging strategies and population dynamics under stoat predation.

## Key Features
- **CEPPM Simulation**: Computes population trajectories based on various predator suppression rates.
- **Tipping Point Analysis**: Identifies the critical harvest threshold and critical intervention times.
- **Strategy Evolution**: Models the shift toward "open foraging" behavior.

## Installation & Usage
This model was implemented in Python using algorithms designed by the author and refined with assistance from Claude.ai (Anthropic)

1. **Clone the repository**: 
   `git clone https://github.com/nts-sms/CEPPM`
2. **Install dependencies**: 
   `pip install numpy scipy matplotlib pandas seaborn`
3. **Run the simulation**:
   - Open and run `CEPPM_2026.ipynb` in a CoLab environment to generate the plots shown in the paper.
   - Open and run `run_scenarios.py` to reproduce the model results for the control and 4 harvest regime scenarios
   - Open and run `sensitivity_all.py` to reproduce the sensitivity analyses as follows: Part 1 — δ and r_stoat sweeps; Part 2 — α and β sweeps; Part 3 — L-V parameter sensitivity (r, α, β, δ); Part 4 — Intervention timing sensitivity for years 0–30
   - Alternatively, open and run `kiwi_conservation_analysis.py` to reproduce all model results - Base model visualisations, all five conservation scenarios,  user-configured harvest rate simulation, Hybrid model baseline and managed comparison, Foraging vs harvest rate sweep, sensitivity analysis and field data analysis



## Author
**Sierra M. Sharma** - Independent Researcher

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
