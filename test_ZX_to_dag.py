import unittest
import numpy as np
from qiskit import QuantumCircuit, execute, Aer, IBMQ, QuantumRegister, ClassicalRegister
from qiskit.transpiler import PassManager
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
import qiskit
from qiskit.tools.visualization import dag_drawer
import pyzx as zx
import fractions
from copy import copy
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler import Layout
from qiskit.circuit.library import SwapGate
from typing import List, Callable, Optional, Union
from qiskit.test import QiskitTestCase
from ZX_to_DAG import zx_to_dag, get_unitary_matrix,matrix_equality
# from Pass_ZX_to_dag import ZX_dag_pass


class TestZXtoDAG(QiskitTestCase):
    """Test for 1q gate optimizations."""

    def test_GHZ(self):
        """Identity gates are like 'wait' commands.
        They should never be optimized (even without barriers).
        See: https://github.com/Qiskit/qiskit-terra/issues/2373
        """
        qc=QuantumCircuit(3)
        qc.h(0)
        qc.cx(0,1)
        qc.cx(0,2)
        
        #convert to zx circuit
        simulator = Aer.get_backend('aer_simulator')
        qc1 = transpile(qc, simulator,optimization_level=0,basis_gates=['u3','cx'])

        g=zx.qasm(qc1.qasm())
        mat1=get_unitary_matrix(qc1)
        d=zx_to_dag(g)
        
        qc2=dag_to_circuit(d)

        mat2=get_unitary_matrix(qc2)
        
        mat1_arr = np.reshape(mat1,mat1.size)
        mat2_arr = np.reshape(mat2,mat2.size)
        first_non_zero_ind = (mat1_arr!=0).argmax(axis=0)
        bool_list=[True]*len(range(first_non_zero_ind))
        possible_scalar = (mat1_arr[first_non_zero_ind])/(mat2_arr[first_non_zero_ind])
        for i in range(mat1_arr.size):
            if (mat1_arr[i]==0 and mat2_arr[i]==0):
                bool_list.append(True)
                print(i)
            if (mat1_arr[i]!=0 and mat2_arr[i]!=0):
                new_sca=(mat1_arr[i])/(mat2_arr[i])
                if (np.isclose(possible_scalar.imag,new_sca.imag,rtol=1e-02) and np.isclose(possible_scalar.real,new_sca.real,rtol=1e-05)):
                    bool_list.append(True)
                    print(i)
        self.assertEqual([True]*mat1.size, bool_list,"it is working")
        
    def test_circ4(self):
        """Identity gates are like 'wait' commands.
        They should never be optimized (even without barriers).
        See: https://github.com/Qiskit/qiskit-terra/issues/2373
        """
        qc=QuantumCircuit(3)
        qc.h(0)
        qc.cx(0,1)
        qc.cx(0,2)
        qc.y(2)
        qc.z(1)
        qc.x(1)
#         qc.cu3(0.2,0.3,0.4,0,1)
        
        #convert to zx circuit
        simulator = Aer.get_backend('aer_simulator')
        qc1 = transpile(qc, simulator,optimization_level=0,basis_gates=['u3','cx'])

        g=zx.qasm(qc1.qasm())
        mat1=get_unitary_matrix(qc1)
        d=zx_to_dag(g)
        
        qc2=dag_to_circuit(d)

        mat2=get_unitary_matrix(qc2)
        
        mat1_arr = np.reshape(mat1,mat1.size)
        mat2_arr = np.reshape(mat2,mat2.size)
        first_non_zero_ind = (mat1_arr!=0).argmax(axis=0)
        bool_list=[True]*len(range(first_non_zero_ind))
        possible_scalar = (mat1_arr[first_non_zero_ind])/(mat2_arr[first_non_zero_ind])
        for i in range(mat1_arr.size):
            if (mat1_arr[i]==0 and mat2_arr[i]==0):
                bool_list.append(True)
                print(i)
            if (mat1_arr[i]!=0 and mat2_arr[i]!=0):
                new_sca=(mat1_arr[i])/(mat2_arr[i])
                if (np.isclose(possible_scalar.imag,new_sca.imag,rtol=1e-20) and np.isclose(possible_scalar.real,new_sca.real,rtol=1e-20)):
                    bool_list.append(True)
                    print(i)
        self.assertEqual([True]*mat1.size, bool_list,"it is working")
        
    def test_circ1(self):
        """Identity gates are like 'wait' commands.
        They should never be optimized (even without barriers).
        See: https://github.com/Qiskit/qiskit-terra/issues/2373
        """
        qc=QuantumCircuit(4)
        qc.ccz(0,1,2)
        qc.h(1)
        qc.t(1)
        qc.ccz(0,1,2)
        qc.h(1)
        qc.t(0)
        qc.ccz(2,1,0)
        qc.s(1)
        qc.ccx(2,1,0)
        qc.crz(0.2,0,1)
        qc.rccx(2,1,0)
        qc.cry(0.4,2,1)
        qc.ch(0,1)
        qc.rcccx(3,1,2,0)

        #convert to zx circuit
        simulator = Aer.get_backend('aer_simulator')
        qc1 = transpile(qc, simulator,optimization_level=0,basis_gates=['u3','cx'])

        g=zx.qasm(qc1.qasm())
        mat1=get_unitary_matrix(qc1)
        d=zx_to_dag(g)

        qc2=dag_to_circuit(d)

#         mat2=get_unitary_matrix(qc2)
        qc2.save_unitary()
        result = simulator.run(qc2).result()
        mat2 = np.asarray(result.get_unitary(qc2))
#         bool_list = matrix_equality(mat1,mat2)

        mat1_arr = np.reshape(mat1,mat1.size)
        mat2_arr = np.reshape(mat2,mat2.size)
        first_non_zero_ind = (mat1_arr!=0).argmax(axis=0)
        bool_list=[True]*len(range(first_non_zero_ind))
        possible_scalar = (mat1_arr[first_non_zero_ind])/(mat2_arr[first_non_zero_ind])
        for i in range(mat1_arr.size):
            if (mat1_arr[i]==0 and mat2_arr[i]==0):
                bool_list.append(True)
            if (mat1_arr[i]!=0 and mat2_arr[i]!=0):
                new_sca=(mat1_arr[i])/(mat2_arr[i])
                if (np.isclose(possible_scalar.imag,new_sca.imag,rtol=1e-15) and np.isclose(possible_scalar.real,new_sca.real,rtol=1e-08)):
                    bool_list.append(True)   

        self.assertEqual([True]*mat1.size, bool_list,"it is working")
        
    def test_circ2(self):
        """Identity gates are like 'wait' commands.
        They should never be optimized (even without barriers).
        See: https://github.com/Qiskit/qiskit-terra/issues/2373
        """
        qc=QuantumCircuit(4)
        qc.swap(0,1)
        qc.swap(1,2)
        qc.ccz(0,1,2)
        qc.h(1)
        qc.t(1)
        qc.ccz(0,1,2)
        qc.h(1)
        qc.t(0)
        qc.ccz(2,1,0)
        qc.cswap(1,2,3)
        qc.s(1)
        qc.ccx(2,1,0)
        qc.crz(0.2,0,1)
        qc.rccx(2,1,0)
        qc.cry(0.4,2,1)
        qc.ch(0,1)
        qc.rcccx(3,1,2,0)

        #convert to zx circuit
        simulator = Aer.get_backend('aer_simulator')
        qc1 = transpile(qc, simulator,optimization_level=0,basis_gates=['u3','cx'])

        g=zx.qasm(qc1.qasm())
        mat1=get_unitary_matrix(qc1)
        d=zx_to_dag(g)

        qc2=dag_to_circuit(d)

        mat2=get_unitary_matrix(qc2)

        mat1_arr = np.reshape(mat1,mat1.size)
        mat2_arr = np.reshape(mat2,mat2.size)
        first_non_zero_ind = (mat1_arr!=0).argmax(axis=0)
        bool_list=[True]*len(range(first_non_zero_ind))
        possible_scalar = (mat1_arr[first_non_zero_ind])/(mat2_arr[first_non_zero_ind])
        for i in range(mat1_arr.size):
            if (mat1_arr[i]==0 and mat2_arr[i]==0):
                bool_list.append(True)
            if (mat1_arr[i]!=0 and mat2_arr[i]!=0):
                new_sca=(mat1_arr[i])/(mat2_arr[i])
                if (np.isclose(possible_scalar.imag,new_sca.imag) and np.isclose(possible_scalar.real,new_sca.real)):
                    bool_list.append(True) 

        self.assertEqual(bool_list,[True]*mat1.size,"it is working") 

if __name__ == "__main__":
    unittest.main()