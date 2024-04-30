# Adaptive Variational Quantum Eigensolver (VQE)

This repository contains a Python script implementing an Adaptive Variational Quantum Eigensolver (VQE) using the PennyLane library. The adaptive VQE aims to efficiently simulate the ground state energy of molecular systems, dynamically adjusting the quantum chemistry basis set based on specified performance criteria.

## Overview

The adaptive VQE algorithm optimizes the ground state energy estimation of a molecular system using a parameterized quantum circuit. This project enhances traditional VQE by incorporating an adaptive mechanism that adjusts the computational basis set based on the results' accuracy and stability. The primary goal is to achieve a high-accuracy simulation with efficient resource utilization.

## Features

- **Molecular Hamiltonian Calculation:** Constructs the Hamiltonian of a molecule using different basis sets.
- **Quantum Circuit Setup:** Initializes and configures a quantum circuit for running VQE.
- **Parameter Optimization:** Utilizes gradient descent to find the minimum energy configuration.
- **Adaptive Mechanism:** Dynamically selects the basis set based on energy and variance thresholds to improve accuracy and stability.

## Requirements

To run this script, you need:
- Python 3.7 or higher
- PennyLane
- Numpy

You can install the required packages using pip:
```bash
pip install pennylane numpy
```

## Usage

To execute the adaptive VQE simulation, simply run the Python script:
```bash
python adaptive_vqe.py
```

### Script Structure

- `create_hamiltonian`: Function to generate the molecular Hamiltonian.
- `setup_vqe_circuit`: Function to setup the VQE circuit.
- `optimize_vqe`: Function to optimize the circuit parameters.
- Main execution block: Sets up and runs the adaptive VQE loop.

## Configuration

Adjust the following parameters in the script to fit your specific simulation needs:
- `symbols`: List of atomic symbols for the molecule.
- `coordinates`: 3D coordinates of the atoms in Angstrom units.
- `basis_sets`: List of basis sets to use for the simulation.
- `max_iterations`: Maximum number of iterations for the adaptive loop.

## Output

The script prints the results of each VQE iteration, showing the basis set used, the calculated energy, and the energy variance. After completing all iterations, it indicates that the adaptive VQE process has finished.

## Contributing

Contributions to this project are welcome! Please fork the repository and submit a pull request with your enhancements.

## License

This project is released under the [MIT License](LICENSE).
