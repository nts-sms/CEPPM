# CEPPM: Coupled Evolutionary Predator-Prey Model for Kiwi Conservation
Repository for Sierra's CEPPM EGT research. This repository contains the source code for the research paper: 
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
   Open and run `CEPPM_2026.ipynb` in a CoLab environment to generate the plots shown in the paper.

## Author
**Sierra M. Sharma** - Independent Researcher

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
