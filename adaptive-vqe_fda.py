import pennylane as qml
from pennylane import numpy as np
import time
import math
import os
import psutil

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

def create_molecule_and_hamiltonian(basis_params):
    # Define custom basis set parameters
    custom_basis = {
        "sto-3g": {
            "H": [basis_params[:2]],
            "Be": [basis_params[2:]]
        }
    }
    
    # Create the Molecule object
    BeH2 = qml.qchem.Molecule(BeHHsymbols, BeHHcoords, charge=BeHHcharge, basis_name="sto-3g")
    
    # Create the Hamiltonian using the custom basis set
    hamiltonian, qubits = qml.qchem.molecular_hamiltonian(
        BeH2.symbols, BeH2.coordinates, charge=BeH2.charge, mult=1, basis="sto-3g", mapping="jordan_wigner"
    )
    
    return BeH2, hamiltonian, qubits

def initialize_vqe(basis_params):
    BeH2, h_vanilla, n_qubits = create_molecule_and_hamiltonian(basis_params)
    n_electrons = BeH2.n_electrons
    hf_state = qml.qchem.hf_state(n_electrons, n_qubits)
    singles, doubles = qml.qchem.excitations(n_electrons, n_qubits)
    dev = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(dev, diff_method='backprop')
    def energy_expval_ASD(params):
        qml.templates.AllSinglesDoubles(params, wires=range(n_qubits), hf_state=hf_state, singles=singles, doubles=doubles)
        return qml.expval(h_vanilla)
    
    return BeH2, h_vanilla, n_qubits, hf_state, singles, doubles, dev, energy_expval_ASD

def compute_gradient_finite_difference(params, basis_params, epsilon=1e-5):
    # Initialize the VQE setup with the current basis parameters
    # This includes creating the molecule, Hamiltonian, and quantum device,
    # and setting up the quantum node for energy calculation.
    BeH2, h_vanilla, n_qubits, hf_state, singles, doubles, dev, energy_expval_ASD = initialize_vqe(basis_params)
    
    # Calculate the energy for the given parameters using the current basis parameters
    base_energy = energy_expval_ASD(params)
    
    # Initialize an empty list to store the gradients for each basis parameter
    gradients = []
    
    # Loop over each basis parameter to compute the gradient
    for i in range(len(basis_params)):
        # Create a copy of the current basis parameters and perturb the i-th parameter positively
        basis_params_plus = basis_params.copy()
        basis_params_plus[i] += epsilon
        
        # Create a copy of the current basis parameters and perturb the i-th parameter negatively
        basis_params_minus = basis_params.copy()
        basis_params_minus[i] -= epsilon
        
        # Reinitialize the VQE setup for the positively perturbed basis parameters
        _, _, _, _, _, _, _, energy_expval_plus = initialize_vqe(basis_params_plus)
        
        # Reinitialize the VQE setup for the negatively perturbed basis parameters
        _, _, _, _, _, _, _, energy_expval_minus = initialize_vqe(basis_params_minus)
        
        # Calculate the energy for the positively perturbed basis parameters
        energy_plus = energy_expval_plus(params)
        
        # Calculate the energy for the negatively perturbed basis parameters
        energy_minus = energy_expval_minus(params)
        
        # Compute the gradient using the finite difference formula
        # (energy_plus - energy_minus) / (2 * epsilon)
        grad = (energy_plus - energy_minus) / (2 * epsilon)
        
        # Append the computed gradient to the list of gradients
        gradients.append(grad)
    
    # Return the list of gradients as a NumPy array
    return np.array(gradients)


def optimize_basis_params(params, basis_params, learning_rate=0.01):
    grad = compute_gradient_finite_difference(params, basis_params)
    return basis_params - learning_rate * grad

def adjust_threshold(energies, current_threshold):
    if len(energies) < 2:
        return current_threshold
    improvement = energies[-2] - energies[-1]
    if improvement < 0.0001:  # Stricter improvement condition
        return current_threshold * 0.95  # Reduce threshold by 5%
    return current_threshold

def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 2)  # Return memory usage in MB

def run_vqe_adaptive(params, basis_params, opt, iterations, initial_threshold, min_iterations_per_basis=10, learning_rate=0.01):
    ti = time.time()
    energies = []
    runtime = []
    energy_threshold = initial_threshold

    global energy_expval_ASD

    lowest_energy = float('inf')
    best_params = None
    best_basis_params = None

    for i in range(iterations):
        t1 = time.time()
        params, energy = opt.step_and_cost(energy_expval_ASD, params)
        t2 = time.time()
        runtime.append(t2 - ti)
        energies.append(energy)
        print(f"Iteration {i + 1}, Energy: {energy} Ha, Memory Usage: {get_memory_usage()} MB")
        
        if energy < lowest_energy:
            lowest_energy = energy
            best_params = params.copy()
            best_basis_params = basis_params.copy()
        
        energy_threshold = adjust_threshold(energies, energy_threshold)
        
        if (i + 1) % min_iterations_per_basis == 0:
            basis_params = optimize_basis_params(params, basis_params, learning_rate)
            BeH2, h_vanilla, n_qubits, hf_state, singles, doubles, dev, energy_expval_ASD = initialize_vqe(basis_params)
            print(f"Updated basis set parameters: {basis_params}")
        
        if len(energies) > 2 and abs(energies[-1] - energies[-2]) < 1e-6:
            print(f"Converged after {i + 1} iterations.")
            break

    print(f"Optimized energy: {lowest_energy} Ha")
    print(f"Best VQE parameters: {best_params}")
    print(f"Best basis set parameters: {best_basis_params}")
    return energies, runtime, best_basis_params

# Initial setup and VQE execution
initial_basis_params = np.array([1.24, 0.5, 0.9, 0.3, 1.2, 0.6], requires_grad=True)
BeH2, h_vanilla, n_qubits, hf_state, singles, doubles, dev, energy_expval_ASD = initialize_vqe(initial_basis_params)

print("\n<Info of VQE with custom basis>")
print("Number of qubits needed:", n_qubits)
print('Number of Pauli strings:', len(h_vanilla.ops))

params_vanilla = np.zeros(len(doubles) + len(singles), requires_grad=True)
adam_opt = qml.AdamOptimizer(stepsize=0.02, beta1=0.9, beta2=0.99, eps=1e-08)

initial_threshold = -15.0
min_iterations_per_basis = 20
energies_vanilla, runtime_vanilla, final_basis_params = run_vqe_adaptive(params_vanilla, initial_basis_params, adam_opt, 200, initial_threshold, min_iterations_per_basis)

print(f"Final optimized basis set parameters: {final_basis_params}")

# Optional: Plot the energy convergence
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(energies_vanilla)
plt.xlabel('Iteration')
plt.ylabel('Energy (Ha)')
plt.title('VQE Energy Convergence with Adaptive Basis Set')
plt.grid(True)
plt.show()
