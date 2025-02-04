from A1 import QuantumCircuit
import numpy as np
from A1 import vectorClose
# test file for tensorizing a gate
def test_x1():
    qc = QuantumCircuit(1, 1)
    qc.x(0)
    (op, q_arr, _) = qc.circuit[qc.pc]
    tensorized = qc.tensorizeGate(op, q_arr)
    matrix = np.array([[0, 1],
                       [1, 0]], 
                      dtype=complex)
    assert(np.allclose(tensorized, matrix))
    print("Test x1 passed")
def test_x2():
    qc = QuantumCircuit(2, 2)
    qc.x(0)
    (op, q_arr, _) = qc.circuit[qc.pc]
    tensorized = qc.tensorizeGate(op, q_arr)
    matrix = np.array([ [0, 0, 1, 0],
                        [0, 0, 0, 1],
                        [1, 0, 0, 0],
                        [0, 1, 0, 0]], 
                      dtype=complex)
    assert(np.allclose(tensorized, matrix))
    print("Test x2 passed")
def test_x3():
    qc = QuantumCircuit(2, 2)
    qc.x(1)
    (op, q_arr, _) = qc.circuit[qc.pc]
    tensorized = qc.tensorizeGate(op, q_arr)
    matrix = np.array([[0, 1, 0, 0],
                       [1, 0, 0, 0],
                       [0, 0, 0, 1],
                       [0, 0, 1, 0]   ], 
                         dtype=complex)
    assert(np.allclose(tensorized, matrix))
    
    print("Test x3 passed")
def test_cx1():
    qc = QuantumCircuit(2, 2)
    qc.cx(0, 1)
    (op, q_arr, _) = qc.circuit[qc.pc]
    tensorized = qc.tensorizeGate(op, q_arr)
    #  # print(tensorized)
    matrix = np.array([ [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 1, 0]], 
                        dtype=complex)
    assert(np.allclose(tensorized, matrix))
    print("Test cx1 passed")
def test_cx2():
    qc = QuantumCircuit(3, 3)
    qc.cx(2, 0)
    (op, q_arr, _) = qc.circuit[qc.pc]
    tensorized = qc.tensorizeGate(op, q_arr)
    #  # print(tensorized)
    matrix = np.array([ [1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0]], 
                            dtype=complex)
    assert(np.allclose(tensorized, matrix))
    print("Test cx2 passed")

def test_cz1():
    qc = QuantumCircuit(2, 2)
    qc.cz(0, 1)
    (op, q_arr, _) = qc.circuit[qc.pc]
    tensorized = qc.tensorizeGate(op, q_arr)
     # print(tensorized)
    matrix = np.array([ [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, -1]], 
                        dtype=complex) 
    assert(np.allclose(tensorized, matrix))
    print("Test cz1 passed")

def test_cz2():
    qc = QuantumCircuit(3, 3)
    qc.cz(0, 1)
    (op, q_arr, _) = qc.circuit[qc.pc]
    tensorized = qc.tensorizeGate(op, q_arr)
     # print(tensorized)
    matrix = np.array([ [1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, -1, 0],
                        [0, 0, 0, 0, 0, 0, 0, -1]], 
                       dtype=complex)
    assert(np.allclose(tensorized, matrix))
    print("Test cz2 passed")

def test_toffoli1():
    qc = QuantumCircuit(3, 3)
    qc.toffoli(0, 1, 2)
    (op, q_arr, _) = qc.circuit[qc.pc]
    tensorized = qc.tensorizeGate(op, q_arr)

    matrix = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0, 0, 1, 0]], 
                      dtype=complex)
    assert(np.allclose(tensorized, matrix))
    print("Test toffoli passed")

def test_s1():
    qc = QuantumCircuit(1, 1)
    qc.s(0)
    (op, q_arr, _) = qc.circuit[qc.pc]
    tensorized = qc.tensorizeGate(op, q_arr)
    matrix = np.array([[1, 0],
                       [0, 1j]], 
                      dtype=complex)
    assert(np.allclose(tensorized, matrix))
    print("Test s1 passed")
def test_s2():
    qc = QuantumCircuit(2, 2)
    qc.s(0)
    (op, q_arr, _) = qc.circuit[qc.pc]
    tensorized = qc.tensorizeGate(op, q_arr)
    matrix = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1j, 0],
                       [0, 0, 0, 1j]], 
                      dtype=complex)
    assert(np.allclose(tensorized, matrix))
    print("Test s2 passed")
def test_s30():
    qc = QuantumCircuit(3, 3)
    qc.s(0)
    (op, q_arr, _) = qc.circuit[qc.pc]
    tensorized = qc.tensorizeGate(op, q_arr)
    matrix = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1j, 0, 0, 0],
    [0, 0, 0, 0, 0, 1j, 0, 0],
    [0, 0, 0, 0, 0, 0, 1j, 0],
    [0, 0, 0, 0, 0, 0, 0, 1j]
], dtype=complex)
    assert(np.allclose(tensorized, matrix))
    print("Test s30 passed")

def test_s31():
    qc = QuantumCircuit(3, 3)
    qc.s(1)
    (op, q_arr, _) = qc.circuit[qc.pc]
    tensorized = qc.tensorizeGate(op, q_arr)
    matrix = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1j, 0, 0, 0, 0, 0],
    [0, 0, 0, 1j, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1j, 0],
    [0, 0, 0, 0, 0, 0, 0, 1j]
], dtype=complex)
    assert(np.allclose(tensorized, matrix))
    print("Test s31 passed")

def test_s32():
    qc = QuantumCircuit(3, 3)
    qc.s(2)
    (op, q_arr, _) = qc.circuit[qc.pc]
    tensorized = qc.tensorizeGate(op, q_arr)
    matrix = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1j, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1j, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1j, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1j]
], dtype=complex)
    assert(np.allclose(tensorized, matrix))
    print("Test s32 passed")

test_x1()
test_x2()
test_x3()
test_cx1()
test_cx2()
test_cz1()
test_cz2()
test_toffoli1()  
test_s1()
test_s2()
test_s30()
test_s31()

print("All tests passed!") 



