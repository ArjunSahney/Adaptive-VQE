import pennylane as qml  # Import PennyLane for quantum computing
from pennylane import numpy as np  # Import numpy from PennyLane for compatibility with auto-differentiation
import time  # Import time module to measure durations
import math  # Import math module for mathematical functions

# Enhanced safe division function with debug output
def safe_divide(a, b, small_number=1e-12):
    # Check if the denominator is very small (less than small_number)
    if np.any(np.abs(b) < small_number):
        print("Warning: small denominator encountered", b)
    # Perform division, substituting small denominators with small_number
    result = np.divide(a, np.where(np.abs(b) < small_number, small_number, b))
    # Check for NaN results and log if found
    if np.any(np.isnan(result)):
        print("NaN produced in division: numerator", a, "denominator", b)
    return result

# Molecular Information
angHBeH = math.pi  # Bond angle for BeH2 molecule, set to 180 degrees (pi radians)
lenBeH = 1.3264  # Bond length for BeH2 molecule in Angstroms
angToBr = safe_divide(1, 0.529177210903)  # Conversion factor from Angstroms to Bohr radii
lenInBr = safe_divide(lenBeH, angToBr)  # Convert bond length from Angstroms to Bohr
cx = lenInBr * math.sin(0.5 * angHBeH)  # X-coordinate for hydrogen atoms in the molecule
cy = lenInBr * math.cos(0.5 * angHBeH)  # Y-coordinate for hydrogen atoms in the molecule
BeHHsymbols = ["Be", "H", "H"]  # List of element symbols in the molecule
BeHHcoords = np.array([[0., cy, 0.], [-cx, 0., 0.], [cx, 0., 0.]])  # 3D coordinates of each atom

# Function to create and return a Hamiltonian
def create_hamiltonian(symbols, coords, charge, basis_name):
    # Initialize a molecule object in PennyLane
    molecule = qml.qchem.Molecule(symbols, coords, charge=charge, basis_name=basis_name)
    try:
        # Generate the molecular Hamiltonian using the specified basis set
        h, n_qubits = qml.qchem.molecular_hamiltonian(molecule.symbols, molecule.coordinates, charge=molecule.charge, basis=basis_name)
    except Exception as e:
        # Handle errors in Hamiltonian creation and log them
        print(f"Failed to create Hamiltonian with basis {basis_name}: {e}")
        return None, 0
    return h, n_qubits

# Quantum Encoding with CISD Ansatz
def circuit_CISD(params, wires):
    # Apply double excitation with first parameter
    qml.DoubleExcitation(params[0], wires=wires[:4])
    # Apply single excitation with second parameter
    qml.SingleExcitation(params[1], wires=wires[2:4])

def energy_expval_CISD(params, h, n_qubits):
    # Handle case where Hamiltonian creation failed by returning infinity
    if n_qubits == 0:
        return float('inf')
    # Define a quantum device for simulation with the appropriate number of qubits
    dev = qml.device("default.qubit", wires=n_qubits)
    @qml.qnode(dev, diff_method='backprop')  # Define a quantum node with backpropagation
    def circuit():
        # Set the initial state of the quantum circuit
        qml.BasisState(np.array([1] * 4 + [0] * (n_qubits - 4)), wires=range(n_qubits))
        # Apply the CISD circuit defined earlier
        circuit_CISD(params, range(n_qubits))
        # Return the expectation value of the Hamiltonian
        return qml.expval(h)
    return circuit()

# Optimizing VQE with Dynamic Convergence Criterion
def run_vqe_adaptive(energy_expval, params, h, n_qubits, opt, initial_threshold):
    ti = time.time()  # Start timing the VQE process
    energies = []  # List to store energy values for each iteration
    convergence_threshold = initial_threshold  # Set the initial convergence threshold
    threshold_decrease_factor = 0.9  # Factor to decrease threshold each iteration if conditions are met
    min_percentage_change = 0.01     # Minimum percentage change in energy to decrease threshold

    # Get initial energy
    prev_energy = energy_expval(params, h, n_qubits)
    energies.append(prev_energy)

    while True:
        # Perform an optimization step and get new parameters and energy
        params, prev_energy = opt.step_and_cost(lambda p: energy_expval(p, h, n_qubits), params)
        energy = energy_expval(params, h, n_qubits)
        energies.append(energy)

        # Check if the change in energy is less than the convergence threshold
        if abs(prev_energy - energy) < convergence_threshold:
            print(f"Convergence reached with energy {energy} Ha at time {time.time() - ti} seconds")
            break

        # Adjust the convergence threshold if the energy change is less than the set percentage
        if abs(prev_energy - energy) < abs(prev_energy) * min_percentage_change:
            convergence_threshold *= threshold_decrease_factor
            print(f"Adjusting convergence threshold to {convergence_threshold}")

    return energies, params

# Adaptive VQE Process with added error handling
def adaptive_vqe_process(init_basis='sto-3g', basis_list=['sto-3g', 'cc-pVDZ', '6-31G']):
    best_energy = float('inf')  # Initialize the best energy to infinity
    best_basis = init_basis  # Start with the initial basis as the best basis
    params = np.zeros(2, requires_grad=True)  # Initialize parameters for the VQE circuit
    adam_opt = qml.AdamOptimizer(stepsize=0.1)  # Initialize the optimizer
    initial_convergence_threshold = 1e-6  # Set the initial convergence threshold

    # Iterate over each basis set in the list
    for basis in basis_list:
        # Create Hamiltonian for the current basis set
        h_vanilla, n_qubits = create_hamiltonian(BeHHsymbols, BeHHcoords, 0, basis)
        if h_vanilla is None:  # Skip this basis set if Hamiltonian creation failed
            continue
        # Run the adaptive VQE and get the energies and optimal parameters
        energies, optimal_params = run_vqe_adaptive(energy_expval_CISD, params, h_vanilla, n_qubits, adam_opt, initial_convergence_threshold)
        min_energy = min(energies)  # Find the minimum energy from the results
        if min_energy < best_energy:  # Update the best energy and basis if the current one is better
            best_energy = min_energy
            best_basis = basis
            params = optimal_params  # Use optimized parameters for next iteration
        print(f"Testing basis set: {basis} with energy {min_energy} Ha")

    print(f"Best performing basis set: {best_basis} with energy {best_energy} Ha")
    return best_energy, best_basis

# Main function to start the adaptive VQE process
if __name__ == '__main__':
    best_energy, best_basis = adaptive_vqe_process()
