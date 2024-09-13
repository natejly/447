####################
# File name: A1.py
# This is the programming portion of Assignment 1
# for CPSC 447/547 Intro to Quantum Computing
#
# Name: 
# NetID:
# Collaborators:
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
        HGate = Gate('h', 1, 1/np.sqrt(2) * np.array([[1,1],[1,-1]], dtype=complex))
        self._append(HGate, [qubit], [])
        return

    # Pauli X gate
    def x(self, qubit):
        return 42

    # Pauli Y gate
    def y(self, qubit):
        return 42

    # Pauli Z gate
    def z(self, qubit):
        return 42

    # Phase gate (sqrt(Z))
    def s(self, qubit):
        return 42

    # S dagger gate (adjoint of S gate)
    def sdg(self, qubit):
        return 42

    # T gate (sqrt(S))
    def t(self, qubit):
        return 42

    # T dagger gate (adjoint of T gate)
    def tdg(self, qubit):
        return 42

    # Controlled X gate (CNOT)
    def cx(self, ctrl_qubit, trgt_qubit):
        return 42

    # Controlled Z gate (CZ)
    def cz(self, ctrl_qubit, trgt_qubit):
        return 42

    # Toffoli gate
    def toffoli(self, ctrl_qubit_1, ctrl_qubit_2, trgt_qubit):
        return 42

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

        # YOUR IMPLEMENTATION HERE

        return gate.matrix

    def evolveOneStep(self):
        # Evolve one step from program counter pc, update state vector
        # Feel free to use/modify this helper function for simulate.
        # OUTPUT: - Return quantum state after one step of evolution
        #         - Update self.curr_state to new quantum state
        #         - Increment program counter pc
        #         - Return None if evolving measurement
        # ASSUME: - Measurements are assumed to be last operations, if any.
        curr_state = self.curr_state
        (op, q_arr, c_arr) = self.circuit[self.pc]
        if op.name != 'measure':
            unitary = self.tensorizeGate(op, q_arr)
            curr_state = unitary @ curr_state
            self.curr_state = curr_state
            self.pc += 1
            return curr_state
        else:
            # print("Already reached end of circuit (excluding measurements).")
            # YOUR IMPLEMENTATION HERE
            # Use sampleBit(p) for random sampling

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
        # YOUR IMPLEMENTATION HERE


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
        q0 = (val >> 0) & 1
        q1 = (val >> 1) & 1
        q2 = 1 if q0 == 1 and q1 == 1 else 0  # Toffoli gate: flip q2 if and only if both q0 and q1 is 1
        qc = QuantumCircuit(n,n)
        if q0 == 1:
            qc.x(0)
        if q1 == 1:
            qc.x(1)
        qc.toffoli(0,1,2)
        qc.simulate()
        qc.measure(list(range(n)),list(range(n)))
        outcome = qc.simulate()
        assert(bool(all(outcome == np.array([bool(q0), bool(q1), bool(q2)]))))
        

####################
# Main tests
####################

def testAll():
    testStateInit()
    testGates()
    testCircuit()
    testHGatesTensorize()
    testHGatesTensorize2()
    testEvolve()
    testSimulate()
    testToffoli()
    # ... You can add more or comment out tests

def main():
    requirement_A1.check()
    testAll()

if __name__ == '__main__':
    main()
