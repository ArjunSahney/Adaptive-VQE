# Import necessary libraries
import pennylane as qml              # Quantum computing library
from pennylane import numpy as np    # NumPy with autodifferentiation support from PennyLane
import time                           # Library for tracking execution time
import math                           # Library for mathematical functions
import os                            # Library for monitoring memory usage
import psutil                        # Library for more detailed system and process information

# Define molecular geometry constants for Beryllium Hydride (BeH2)
angHBeH = math.pi                     # Bond angle in radians (180 degrees for linear BeH2)
lenBeH = 1.3264                       # Bond length in Angstroms
angToBr = 1 / 0.529177210903          # Conversion factor from Angstroms to Bohr radii

# Convert bond length from Angstroms to Bohr
lenInBr = lenBeH * angToBr            
cx = lenInBr * math.sin(0.5 * angHBeH) # x-coordinate for hydrogen atoms
cy = lenInBr * (math.cos(0.5 * angHBeH) + 1 - 1) # y-coordinate for Be atom, simplified calculation

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

# Function to initialize VQE parameters and states
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

# Initial basis set
basis_sets = ['sto-3g', 'cc-pVDZ']  # Simplified basis sets for initial testing
initial_basis = basis_sets[0]
BeH2, h_vanilla, n_qubits, hf_state, singles, doubles, dev, energy_expval_ASD = initialize_vqe(initial_basis)

# Print out basic information about the VQE setup
print("\n<Info of vanilla VQE>")
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

    global energy_expval_ASD  # Ensure energy_expval_ASD is accessible inside the function

    BeH2, h_vanilla, n_qubits, hf_state, singles, doubles, dev = [None] * 7

    for i in range(iterations):
        t1 = time.time()
        params, energy = opt.step_and_cost(energy_expval_ASD, params)
        t2 = time.time()
        runtime.append(t2 - ti)
        energies.append(energy)
        print(f"Iteration {i + 1}, Energy: {energy} Ha, Memory Usage: {get_memory_usage()} MB")
        if (i + 1) % 5 == 0:
            print(f"Completed iteration: {i + 1}")
            print(f"Energy: {energy} Ha")
            print("Step Time:", t2 - t1, "s")
            print("----------------")
        if energy_threshold and energy < energy_threshold:
            print(f"Energy threshold reached: {energy} Ha")
            if current_basis_index < len(basis_sets) - 1:
                current_basis_index += 1
                new_basis = basis_sets[current_basis_index]
                print(f"Switching to new basis set: {new_basis}")
                # Free memory by deleting the old objects
                if BeH2 is not None:
                    del BeH2, h_vanilla, n_qubits, hf_state, singles, doubles, dev
                    import gc
                    gc.collect()
                BeH2, h_vanilla, n_qubits, hf_state, singles, doubles, dev, energy_expval_ASD = initialize_vqe(new_basis)
                # Reinitialize the parameters with the correct shape
                params = np.zeros(len(singles) + len(doubles), requires_grad=True)
                energy_threshold /= 10  # Adjust threshold for next stage if needed
            else:
                print("No more basis sets to switch to.")
                break
    print(f"Optimized energy: {energy} Ha")
    return energies, runtime, basis_sets[current_basis_index]

# Initialize parameters for the VQE circuit
params_vanilla = np.zeros(len(doubles) + len(singles), requires_grad=True)

# Setup the optimizer (Adam) with specified hyperparameters
adam_opt = qml.AdamOptimizer(stepsize=0.02, beta1=0.9, beta2=0.99, eps=1e-08)

# Execute the VQE algorithm and capture energies and runtimes
adaptive_basis_threshold = 1e-3  # Energy threshold for switching basis sets
energies_vanilla, runtime_vanilla, final_basis = run_vqe_adaptive(params_vanilla, adam_opt, 20, basis_sets, energy_threshold=adaptive_basis_threshold)

print(f"Final basis set used: {final_basis}")

