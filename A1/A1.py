####################
# File name: A1.py
# This is the programming portion of Assignment 1
# for CPSC 447/547 Intro to Quantum Computing
#
# Name: Nate Ly
# NetID: njl36
# Collaborators: Alvin Lu for the optional part 
# Sources:
# https://www.youtube.com/watch?v=0ECbWBBbglw
# https://quantumcomputing.stackexchange.com/questions/4252/how-to-derive-the-cnot-matrix-for-a-3-qubit-system-where-the-control-target-qu
# https://quantumcomputing.stackexchange.com/questions/18044/how-the-single-qubit-unitary-u-calculates-when-apply-a-gate-to-only-one-qubit
####################

import requirement_A1
import math
import numpy as np
####################
# Helper functions
####################

np.random.seed(447547) 

def complexClose(d1, d2, epsilon=10**-7):
    return (abs(d1.real - d2.real) < epsilon) and (abs(d1.imag - d2.imag) < epsilon) 

def vectorClose(v1, v2, epsilon=10**-7):
    return len(v1) == len(v2) and bool(all([complexClose(v1[i], v2[i], epsilon=10**-7) for i in range(len(v1))]))

def matrixClose(m1, m2, epsilon=10**-7):
    return np.allclose(m1, m2, atol=epsilon)

def sampleBit(p):
    r = np.random.random_sample()
    if r < p:
        return bool(0)
    else:
        return bool(1)

####################
# Place for your own helper functions, if any
####################



####################
# Building Blocks
####################

class Qubit(object):
    """Qubit object"""
    def __init__(self, arg, label='q'):
        super(Qubit, self).__init__()
        self.arg = arg
        self.label = label

class QuantumRegister(object):
    """QuantumRegister is where we keep track of qubits"""
    def __init__(self, num_q, label='qreg'):
        super(QuantumRegister, self).__init__()
        self.size = num_q
        self.label = label
        self.array = [Qubit(i) for i in range(num_q)]
        self.state = np.array([1] + [0] * (2 ** num_q - 1), dtype=complex)

class ClassicalRegister(object):
    """ClassicalRegister is where we keep track of measurement outcomes"""
    def __init__(self, num_c, label='creg'):
        super(ClassicalRegister, self).__init__()
        self.size = num_c
        self.label = label
        self.state = np.array([bool(0) for _ in range(num_c)])
        
class Gate(object):
    """Gate object to describe its name, kind, and matrix"""
    def __init__(self, name, num_q, matrix):
        super(Gate, self).__init__()
        self.name = name 
        self.num_q = num_q
        self.matrix = matrix
        
####################
# In-House Quantum Circuit
####################

class QuantumCircuit(object):
    """QuantumCircuit"""
    def __init__(self, num_q, num_c):
        super(QuantumCircuit, self).__init__()
        self.num_q = num_q
        self.qubits = QuantumRegister(num_q) # initialized qubits
        self.num_c = num_c
        self.cbits = ClassicalRegister(num_c) # initialized cbits
        self.circuit = [] # sequence of instructions
        self.pc = 0 # program counter
        self.curr_state = self.qubits.state # state up to the point of program counter

    def _append(self, operation, q_array, c_array):
        # Add new instruction to circuit
        instruction = [operation, q_array, c_array]
        self.circuit.append(instruction)

    def __repr__(self):
        # For displaying quantum circuit
        qasm = ['\n======<CPSC 447/547 QASM>======']
        qasm += ['Qreg: %d, Creg: %d' % (self.num_q, self.num_c)]
        for inst in self.circuit:
            (op, q_arr, c_arr) = inst
            inst_str = '%s ' % op.name 
            for q in q_arr:
                qubit = self.qubits.array[q]
                inst_str += '%s%d ' % (qubit.label, qubit.arg)
            inst_str += ', '
            for c in c_arr:
                inst_str += '%s%d ' % (self.cbits.label, c)
            qasm.append(inst_str)
        qasm.append('===============================\n')
        return "\n".join(qasm)

    ####################
    # Define instruction set
    # YOUR IMPLEMENTATION HERE
    ####################
    # Hadamard gate
    def h(self, qubit):
        # Define Gate by its name, kind (number of qubit), and matrix
        HGate = Gate('h', 1, 1/np.sqrt(2) * np.array([[1,1],
                                                      [1,-1]], dtype=complex))
        self._append(HGate, [qubit], [])
        return

    # Pauli X gate
    def x(self, qubit):
        XGate = Gate('x', 1, np.array([[0, 1], 
                                       [1, 0]], dtype=complex))
        self._append(XGate, [qubit], [])
        return

    # Pauli Y gate
    def y(self, qubit):
        # j is imaginary unit
        YGate = Gate('y', 1, np.array([[0, -1j], 
                                       [1j, 0]], 
                                      dtype=complex))
        self._append(YGate, [qubit], [])
        return

    # Pauli Z gate
    def z(self, qubit):
        ZGate = Gate('z', 1, np.array([[1, 0], 
                                       [0, -1]], 
                                      dtype=complex))
        self._append(ZGate, [qubit], [])
        return

    # Phase gate (sqrt(Z))
    def s(self, qubit):
        SGate = Gate('s', 1, np.array([[1, 0], 
                                       [0, 1j]], 
                                      dtype=complex))
        self._append(SGate, [qubit], [])
        return

    # S dagger gate (adjoint of S gate)
    def sdg(self, qubit):
        SDGGate = Gate('sdg', 1, np.array([[1, 0], 
                                           [0, -1j]], 
                                          dtype=complex))
        self._append(SDGGate, [qubit], [])

    # T gate (sqrt(S))
    def t(self, qubit):
        e = np.exp((np.pi * 1j) / 4)
        TGate = Gate('t', 1, np.array([[1, 0], 
                                       [0, e]], 
                                      dtype=complex))
        self._append(TGate, [qubit], [])
        return

    # T dagger gate (adjoint of T gate)
    def tdg(self, qubit):
        e = np.exp((-np.pi * 1j) / 4)
        TDGGate = Gate('tdg', 1, np.array([[1, 0], 
                                           [0, e]], 
                                          dtype=complex))
        self._append(TDGGate, [qubit], [])
        return

    # Controlled X gate (CNOT)
    def cx(self, ctrl_qubit, trgt_qubit):
        n = self.num_q
        # Initialize a 2^n x 2^n matrix filled with zeros
        matrix = np.zeros((2**n, 2**n), dtype=complex)
        # Create all possible combinations of n bits for basis
        combos = []
        for i in range(2**n):
            binary = format(i, f'0{n}b')
            combination = [int(b) for b in binary]
            combos.append(combination)
        # print(f"combos1: {combos}")
        # get the combos with cx
        combos2 = []
        for combo in combos:
            combo2 = combo.copy()
            if combo2[ctrl_qubit] == 1:
                combo2[trgt_qubit] = (combo2[trgt_qubit] + 1) % 2
            combos2.append(combo2)
        # print(f"combos2: {combos2}")
        for i, combo in enumerate(combos):
            j = combos2.index(combo)
            k = combos.index(combo)
            matrix[j, k] = 1
        # print(f"matrix: \n {matrix}")

        CXGate = Gate('cx', self.num_q, matrix)
        self._append(CXGate, [ctrl_qubit, trgt_qubit], [])
    # Controlled Z gate (CZ)
    def cz(self, ctrl_qubit, trgt_qubit):
        n = self.num_q
        matrix = np.zeros((2**n, 2**self.num_q), dtype=complex)
        # flip the sign of the state when both qubits are 1
        combos = []
        for i in range(2**self.num_q):
            binary = format(i, f'0{self.num_q}b')
            combination = [int(b) for b in binary]
            combos.append(combination)
        
        combos2 = []
        for combo in combos:
            # transform the combo based on the control and target qubits in cz gate
            combo2 = combo.copy()
            if combo2[ctrl_qubit] == 1 and combo2[trgt_qubit] == 1:
                combo2[trgt_qubit] = -combo2[trgt_qubit]
            combos2.append(combo2)
        # print(f"combos: {combos}")
        # print(f"combos2: {combos2}")
        
        for i in range (len(combos)):
            if combos[i] == combos2[i]:
                matrix[i][i] = 1
            else:
                matrix[i][i] = -1            
    
        CZGate = Gate('cz', self.num_q, matrix)
        self._append(CZGate, [ctrl_qubit, trgt_qubit], [])
    
    # Toffoli gate
    def toffoli(self, ctrl_qubit_1, ctrl_qubit_2, trgt_qubit):
        # flips the target qubit if both control qubits are 1
        n = self.num_q
        if n < 3:
            raise ValueError("Toffoli gate requires at least 3 qubits.")
        matrix = np.zeros((2**n, 2**n), dtype=complex)
        # Create all possible combinations of n bits for basis
        combos = []
        for i in range(2**n):
            binary = format(i, f'0{n}b')
            combination = [int(b) for b in binary]
            combos.append(combination)
        # print(f"combos1: {combos}")
        # get the combos with 
        combos2 = []
        for combo in combos:
            combo2 = combo.copy()
            if combo2[ctrl_qubit_1] == 1 and combo2[ctrl_qubit_2] == 1:
                combo2[trgt_qubit] = (combo2[trgt_qubit] + 1) % 2
            combos2.append(combo2)
        # print(f"combos2: {combos2}")
        for i, combo in enumerate(combos):
            j = combos2.index(combo)
            k = combos.index(combo)
            matrix[j, k] = 1
        # print(f"matrix: \n {matrix}")
        
        ToffoliGate = Gate('toffoli', n, matrix)
        self._append(ToffoliGate, [ctrl_qubit_1, ctrl_qubit_2, trgt_qubit], [])
        return

    # Measure qubits in array 'qubits' and store classical outcome in 'cbits'
    # Note: Action of measurement will be defined in simulate function. 
    def measure(self, qubits, cbits):
        assert(len(qubits) == len(cbits))
        Measure = Gate('measure', len(qubits), None)
        self._append(Measure, qubits, cbits)
        return

    ####################
    # In-House State Vector Simulator
    ####################
    def tensorizeGate(self, gate, q_arr):
            # A function for extend single gate to 2^n by 2^n unitary matrix
            # OUTPUT: - A unitary matrix of size 2^n by 2^n, where n is self.num_q
            #         - return None if invalid input
            # ASSUME: - gate applies to qubits in q_arr
            #         - in q_arr, the first qubit is the most significant bit (MSB)
            #         - number of qubits in q_arr matches gate.num_q
            #         - need to check if gate is not measure
            if gate.name in ['cx', 'cz', 'toffoli']:
                return gate.matrix
            if gate.num_q != len(q_arr) or gate.name == 'measure':
                return None

            n = self.num_q 
            full_matrix = np.eye(1, dtype=complex)  # Initialize as a 1x1 matrix to build upon
                    
            # Iterate through all qubits in the system
            for i in range(n):
                # If the current qubit index is in q_arr, apply the gate's matrix
                if i in q_arr:
                    # Find the position of the qubit in q_arr and apply the gate matrix
                    full_matrix = np.kron(full_matrix, gate.matrix)
                else:
                    # Apply identity matrix for qubits not in q_arr
                    full_matrix = np.kron(full_matrix, np.eye(2, dtype=complex))
            
            return full_matrix

    def domeasure(self, qubits, cbits):
        r = np.random.random_sample()
        prob = 0.0
        measured_state = -1

        for i, amplitude in enumerate(self.curr_state):
            prob += np.abs(amplitude)**2
            if r < prob:
                measured_state = i
                break

        for i in range(len(cbits)):
            cbits[len(cbits) - i - 1] = bool(measured_state & 1)
            measured_state >>= 1

    def evolveOneStep(self):
        """Returns none if measurement is reached"""
        # Evolve one step from program counter pc, update state vector
        # Feel free to use/modify this helper function for simulate.
        # OUTPUT: - Return quantum state after one step of evolution
        #         - Update self.curr_state to new quantum state
        #         - Increment program counter pc
        #         - Return None if evolving measurement
        # ASSUME: - Measurements are assumed to be last operations, if any.
        curr_state = self.curr_state
        # opperation , qubit array, classical array
        (op, q_arr, c_arr) = self.circuit[self.pc]
        if op.name != 'measure':
            unitary = self.tensorizeGate(op, q_arr)
            # matrix multiplication
            # print unitary and curr_state
            # print(f"unitary: {unitary}")
            # print(f"curr_state: {curr_state}")
            curr_state = unitary @ curr_state
            self.curr_state = curr_state
            self.pc += 1
            return curr_state
        else:
            self.domeasure(q_arr, self.cbits.state)
            self.pc += 1
            return None


    def simulate(self):
        # A function for simulating circuit from scratch (from pc = 0).
        # OUTPUT: - If circuit has measure, return outcome stored in cbits
        #         - Otherwise, return the state of the qubits in qreg
        #         - Program counter (pc) reaches the end of circuit
        # ASSUME: - Measurements are assumed to be last operations, if any.
        #         - Start with initial all-zero state in self.qubits.state
        measured = False
        curr_state = self.qubits.state # initial state
        while self.pc < len(self.circuit):
            # evolve one step
            curr_state = self.evolveOneStep()
            if curr_state is None:
                measured = True
                break

        self.qubits.state = curr_state # final state
        if measured:
            return self.cbits.state
        else:
            return self.qubits.state
                 


####################
# Test functions (feel free to add more of your own tests here)
####################
def testStateInit():
    qreg1 = QuantumRegister(1)
    assert(vectorClose(qreg1.state, np.array([1, 0], dtype=complex)))
    qreg2 = QuantumRegister(2)
    assert(vectorClose(qreg2.state, np.array([1, 0, 0, 0], dtype=complex)))
    qc = QuantumCircuit(2,2)
    assert(vectorClose(qc.qubits.state, np.array([1, 0, 0, 0], dtype=complex)))

def testGates():
    qc = QuantumCircuit(2,2)
    qc.h(0)
    assert(matrixClose(qc.circuit[0][0].matrix, 1/np.sqrt(2) * np.array([[1,1],[1,-1]], dtype=complex)))

def testCircuit():
    qc = QuantumCircuit(2,2)
    qc.h(0)
    # print(qc)

def testHGatesTensorize():
    qc = QuantumCircuit(2,2)
    qc.h(0)
    (op, q_arr, _) = qc.circuit[qc.pc]
    tensorized = qc.tensorizeGate(op, q_arr)
    assert(matrixClose(tensorized, np.array([
        [ 0.70710678+0.j,  0.        +0.j,  0.70710678+0.j,  0.        +0.j],
        [ 0.        +0.j,  0.70710678+0.j,  0.        +0.j,  0.70710678+0.j],
        [ 0.70710678+0.j,  0.        +0.j, -0.70710678+0.j,  0.        +0.j],
        [ 0.        +0.j,  0.70710678+0.j,  0.        +0.j, -0.70710678+0.j]
    ], dtype=complex)))

def testHGatesTensorize2():
    qc = QuantumCircuit(2,2)
    qc.h(1)
    (op, q_arr, _) = qc.circuit[qc.pc]
    tensorized = qc.tensorizeGate(op, q_arr)
    assert(matrixClose(tensorized, np.array([
        [ 0.70710678+0.j,  0.70710678+0.j,  0.        +0.j,  0.        +0.j],
        [ 0.70710678+0.j, -0.70710678+0.j,  0.        +0.j,  0.        +0.j],
        [ 0.        +0.j,  0.        +0.j,  0.70710678+0.j,  0.70710678+0.j],
        [ 0.        +0.j,  0.        +0.j,  0.70710678+0.j, -0.70710678+0.j]
    ], dtype=complex)))

def testEvolve():
    qc = QuantumCircuit(1,1)
    qc.h(0)
    qc.evolveOneStep()
    assert(vectorClose(qc.curr_state, np.array([[1/np.sqrt(2)], [1/np.sqrt(2)]], dtype=complex)))

def testSimulate():
    qc = QuantumCircuit(1,1)
    qc.h(0)
    qc.h(0)
    qc.measure([0],[0])
    outcome = qc.simulate()
    assert(bool(all(outcome == np.array([bool(0)]))))

def testToffoli():
    n = 3
    for val in range(4):
        # print(f"val: {val}")
        q0 = (val >> 0) & 1
        q1 = (val >> 1) & 1
        q2 = 1 if q0 == 1 and q1 == 1 else 0  # Toffoli gate: flip q2 if and only if both q0 and q1 is 1
        qc = QuantumCircuit(n,n)
        if q0 == 1:
            qc.x(0)
        if q1 == 1:
            qc.x(1)
        qc.toffoli(0,1,2)
        # breaking here 
        qc.simulate()
        qc.measure(list(range(n)),list(range(n)))
        outcome = qc.simulate()
        # print(f"outcome: {outcome}")
        # print(f"expected: {np.array([bool(q0), bool(q1), bool(q2)])}")
        assert(bool(all(outcome == np.array([bool(q0), bool(q1), bool(q2)]))))
    

####################
# Main tests
####################

def testAll():
    testStateInit()
    print("Passed testStateInit")
    testGates()
    print("Passed testGates")
    testCircuit()
    print("Passed testCircuit")
    testHGatesTensorize()
    print("Passed testHGatesTensorize")
    testHGatesTensorize2()
    print("Passed testHGatesTensorize2")
    testEvolve()
    print("Passed testEvolve")
    testSimulate()
    print("Passed testSimulate")
    testToffoli()
    print("Passed testToffoli")
    # ... You can add more or comment out tests

def main():
    requirement_A1.check()
    print("requirenment check passed")
    testAll()

if __name__ == '__main__':
    main()
