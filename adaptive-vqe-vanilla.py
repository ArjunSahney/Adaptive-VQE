import pennylane as qml
from pennylane import numpy as np
import time
import math
import os
import psutil
import gc

# Define molecular geometry constants for Beryllium Hydride (BeH2)
angHBeH = math.pi  # Bond angle in radians (180 degrees for linear BeH2)
lenBeH = 1.3264  # Bond length in Angstroms
angToBr = 1 / 0.529177210903  # Conversion factor from Angstroms to Bohr radii

# Convert bond length from Angstroms to Bohr
lenInBr = lenBeH * angToBr
cx = lenInBr * math.sin(0.5 * angHBeH)  # x-coordinate for hydrogen atoms
cy = lenInBr * math.cos(0.5 * angHBeH)  # y-coordinate for Be atom

# Calculate the distance between the two hydrogen atoms
lenHH = 2 * cx

# Define the symbols and coordinates for the nuclei in BeH2
BeHHsymbols = ["Be", "H", "H"]
BeHHcoords = np.array([[0., cy, 0.], [-cx, 0., 0.], [cx, 0., 0.]])

# Define the net charge of the molecule
BeHHcharge = 0

# Function to create Molecule object and Hamiltonian in PennyLane with a predefined basis set
def create_molecule_and_hamiltonian(basis_name):
    BeH2 = qml.qchem.Molecule(BeHHsymbols, BeHHcoords, charge=BeHHcharge, basis_name=basis_name)
    hamiltonian, n_qubits = qml.qchem.molecular_hamiltonian(
        BeH2.symbols, BeH2.coordinates, charge=BeH2.charge, basis=BeH2.basis_name, name=f"BeH2_{basis_name}"
    )
    return BeH2, hamiltonian, n_qubits

# Function to initialize VQE parameters and states with a given basis set
def initialize_vqe(basis_name):
    BeH2, h_vanilla, n_qubits = create_molecule_and_hamiltonian(basis_name)
    n_electrons = BeH2.n_electrons
    hf_state = qml.qchem.hf_state(n_electrons, n_qubits)
    singles, doubles = qml.qchem.excitations(n_electrons, n_qubits)
    dev = qml.device("default.qubit", wires=h_vanilla.wires)
    
    @qml.qnode(dev, diff_method='backprop')
    def energy_expval_ASD(params):
        qml.templates.AllSinglesDoubles(params, wires=h_vanilla.wires, hf_state=hf_state, singles=singles, doubles=doubles)
        return qml.expval(h_vanilla)
    
    return BeH2, h_vanilla, n_qubits, hf_state, singles, doubles, dev, energy_expval_ASD

# Define initial basis sets for adaptive VQE
basis_sets = ['sto-3g', '6-31g', '6-311g', 'cc-pvdz']

# Initialize with the smallest basis set 'sto-3g'
BeH2, h_vanilla, n_qubits, hf_state, singles, doubles, dev, energy_expval_ASD = initialize_vqe('sto-3g')

# Print out basic information about the VQE setup
print("\n<Info of VQE with smallest basis>")
print("Number of qubits needed:", n_qubits)
print('Number of Pauli strings:', len(h_vanilla.ops))

# Function to monitor memory usage
def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 2)  # Return memory usage in MB

# Function to run the VQE algorithm with adaptive basis sets
def run_vqe_adaptive(params, opt, iterations, basis_sets, energy_threshold):
    ti = time.time()
    energies = []
    runtime = []
    current_basis_index = 0
    last_energy = float('inf')

    global energy_expval_ASD

    for i in range(iterations):
        t1 = time.time()
        params, energy = opt.step_and_cost(energy_expval_ASD, params)
        t2 = time.time()
        runtime.append(t2 - ti)
        energies.append(energy)
        improvement = last_energy - energy
        print(f"Iteration {i + 1}, Energy: {energy} Ha, Improvement: {improvement}, Memory Usage: {get_memory_usage()} MB")
        
        last_energy = energy
        
        if (i + 1) % 5 == 0 or energy < energy_threshold or improvement < 0.001:
            if current_basis_index < len(basis_sets) - 1 and (improvement < 0.001 or energy < energy_threshold):
                current_basis_index += 1
                new_basis = basis_sets[current_basis_index]
                print(f"Switching to new basis set: {new_basis} due to insufficient improvement or reaching the threshold.")
                BeH2, h_vanilla, n_qubits, hf_state, singles, doubles, dev, energy_expval_ASD = initialize_vqe(new_basis)
                params = np.zeros(len(singles) + len(doubles), requires_grad=True)
            else:
                print("No more basis sets to switch to, or improvement is satisfactory.")
                break

    print(f"Optimized energy: {energy} Ha")
    return energies, runtime, basis_sets[current_basis_index]

# Initialize parameters for the VQE circuit
params_vanilla = np.zeros(len(doubles) + len(singles), requires_grad=True)

# Setup the optimizer (Adam) with specified hyperparameters
adam_opt = qml.AdamOptimizer(stepsize=0.02, beta1=0.9, beta2=0.99, eps=1e-08)

# Execute the VQE algorithm and capture energies and runtimes
adaptive_basis_threshold = -14.5  # Energy threshold for switching basis sets
energies_vanilla, runtime_vanilla, final_basis = run_vqe_adaptive(params_vanilla, adam_opt, 100, basis_sets, adaptive_basis_threshold)

print(f"Final basis set used: {final_basis}")
