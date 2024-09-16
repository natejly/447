from A1 import QuantumCircuit
import numpy as np
# test file for tensorizing a gate
def test_cx():
    qc = QuantumCircuit(3, 3)
    qc.cx(2, 0)
    (op, q_arr, _) = qc.circuit[qc.pc]
    tensorized = qc.tensorizeGate(op, q_arr)
    print("\n")
    print(tensorized)
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
test_cx()



