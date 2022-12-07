from qiskit import QuantumCircuit, execute, Aer, IBMQ, QuantumRegister, ClassicalRegister
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
import numpy as np
import graphviz
import qiskit
from qiskit.converters import circuit_to_dag
from qiskit.tools.visualization import dag_drawer
import pyzx as zx
import fractions
from copy import copy
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler import Layout
from qiskit.circuit.library import SwapGate
from typing import List, Callable, Optional, Union

def _gf_to_circ(gf):
    inp = gf.inputs()
    out=gf.outputs()
    qmap=gf.qubits() 
    nqubits=len(list(set(list(qmap.values()))))
    qmap_inp = [qmap.get(i) for i in inp]
    qmap_out = [qmap.get(i) for i in out]
    only_inp_q = [q for q in qmap_inp if q not in qmap_out]
    only_out_q = [q for q in qmap_out if q not in qmap_inp]
    qc=QuantumCircuit(nqubits)
    if len(only_inp_q+only_out_q)!=0:
        qc.h(only_inp_q+only_out_q)
    ed_gen=[]
    a=gf.edges()
    for i in a:
        ed_gen.append(i)
    for edge in ed_gen:
        if (gf.qubit(edge[0])==gf.qubit(edge[1])):              
            if (isinstance(gf.phase(edge[0]),fractions.Fraction)):
                ph=gf.phase(edge[0])
                qc.rz(ph*np.pi,gf.qubit(edge[0]))
            if (gf.edge_type(edge)==2):
                qc.h(gf.qubit(edge[0]))
            if (isinstance(gf.phase(edge[1]),fractions.Fraction)):
                ph=gf.phase(edge[1])
                qc.rz(ph*np.pi,gf.qubit(edge[1]))
        if (gf.qubit(edge[0])!=gf.qubit(edge[1])):
            if (isinstance(gf.phase(edge[0]),fractions.Fraction)):
                ph=gf.phase(edge[0])
                qc.rz(ph*np.pi,gf.qubit(edge[0]))
            if (gf.edge_type(edge)==2):
                qc.cz(gf.qubit(edge[0]),gf.qubit(edge[1]))
            if (gf.edge_type(edge)==1):
                qc.cx(gf.qubit(edge[0]),gf.qubit(edge[1]))
            if (isinstance(gf.phase(edge[1]),fractions.Fraction)):
                ph=gf.phase(edge[1])
                qc.rz(ph*np.pi,gf.qubit(edge[1]))
    return qc


def _zxcirc_to_qiskit_circ(c2):
    n=c2.qubits
    qc=QuantumCircuit(n)
    for g in c2.gates:
        print(g)
        if g.name == 'SWAP':
            qc.swap(g.control,g.target)
        if g.name == 'HAD':
            qc.h(g.target)
        if g.name == 'ZPhase':
            qc.rz(g.phase*np.pi,g.target)
        if g.name == 'CZ':
            qc.cz(g.control, g.target)
        if g.name == 'CNOT':
            qc.cx(g.control, g.target)
        if g.name == 'S':
            qc.rz(g.phase*np.pi, g.target)
        if g.name == 'T':
            qc.rz(g.phase*np.pi, g.target)
        if g.name == 'XPhase':
            qc.rx(g.phase*np.pi, g.target) 
        if g.name == 'CCZ':
            qc.ccz(g.ctrl1,g.ctrl2,g.target)
        if g.name=='CHAD':
            qc.ch(g.control,g.target)
        if g.name == 'CRZ':
            qc.crz(g.control,g.target,g.phase)
        if g.name=='Tof':
            qc.ccx(g.ctrl1,g.ctrl2,g.target)     
        
    return qc


def clone(gr):
    cpy = zx.Graph()
    for v, d in gr.graph.items():
        cpy.graph[v] = d.copy()
        cpy._vindex = gr._vindex
        cpy.nedges = gr.nedges
        cpy.ty = gr.ty.copy()
        cpy._phase = gr._phase.copy()
        cpy._qindex = gr._qindex.copy()
        cpy._maxq = gr._maxq
        cpy._rindex = gr._rindex.copy()
        cpy._maxr = gr._maxr
        cpy._vdata = gr._vdata.copy()
        cpy.scalar = gr.scalar.copy()
        cpy._inputs = gr._inputs
        cpy._outputs = gr._outputs
        cpy.track_phases = gr.track_phases
        cpy.phase_index = gr.phase_index.copy()
        cpy.phase_master = gr.phase_master
        cpy.phase_mult = gr.phase_mult.copy()
        cpy.max_phase_index = gr.max_phase_index
    return cpy


def zx_to_dag(g):
    if (isinstance(g, zx.circuit.Circuit)):
#         g1=g.to_basic_gates()
        g1 = g.to_graph()
#         zx.simplify.full_reduce(g1)
        zx.simplify.full_reduce(g1)
        g1 = zx.extract_circuit(g1).to_basic_gates()
#         g1 = zx.extract_circuit(g1)
#         qc=_zxcirc_to_qiskit_circ(g1)  
#         dag = circuit_to_dag(qc)
#         return dag
        try:
            g2 = zx.optimize.full_optimize(g1)
        except:
            print('y1')
            qc=_zxcirc_to_qiskit_circ(g1)  
            dag = circuit_to_dag(qc)
            return dag
        else:
            print('y2')
            qc=_zxcirc_to_qiskit_circ(g2)  
            dag = circuit_to_dag(qc)
            return dag
    if isinstance(g,zx.graph.graph_s.GraphS):
        for i in g.qubits().values():
                if isinstance(i,float):
                    print('''for zx graphs, 3 qubit gates and other multi-qubit gates(beyond 2 qubits) are not supported right now.\n Most                              multi-qubit gates can be decomposed into 2 qubits and single qubit gates. We recommend \ndecomposing                                        these gates.''')
                    return
                    break;   
        try:
            new_circ=zx.extract_circuit(g).to_basic_gates()            
        except:
            print('no direct extraction')
            g2=clone(g)
            zx.full_reduce(g2)
            # g2.normalize()
            qmap=g2.qubits()
            if (sum(1 for num in list(qmap.values()) if num < 0)==0):
                print('case 1')
                qc= _gf_to_circ(g2)
                dag = circuit_to_dag(qc)
                return dag
            if (sum(1 for num in list(qmap.values()) if num < 0)>0):
                print('case 2')
                zx.to_gh(g)
                qc= _gf_to_circ(g)
                dag = circuit_to_dag(qc)
                return dag  
        else:
            print('case 3')
            try:
                g2 = zx.optimize.full_optimize(g1)
            except:
                qc=_zxcirc_to_qiskit_circ(g1)  
                dag = circuit_to_dag(qc)
                return dag
            else:
                qc=_zxcirc_to_qiskit_circ(g2)  
                dag = circuit_to_dag(qc)
                return dag
            
            
def get_unitary_matrix(qc):
    simulator = Aer.get_backend('aer_simulator')
    qc1 = transpile(qc, simulator,optimization_level=0,basis_gates=['u3','cx'])
    qc1.save_unitary()
    result = simulator.run(qc1).result()
    unitary = np.asarray(result.get_unitary(qc1))
    return unitary



def matrix_equality(mat1,mat2):
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
    return bool_list
            


class ZXSwap(TransformationPass):
    """converts from ZX diagrams to Qiskit DAG circuits and DAGs to ZX diagrams"""

    def __init__(self):
        
        super().__init__()

    def run(self, obj: Union[qiskit.circuit.quantumcircuit.QuantumCircuit,zx.graph.graph_s.GraphS, zx.circuit.Circuit,
                            qiskit.dagcircuit.dagcircuit.DAGCircuit]):
        """
        Args:
            obj: Could be ZX circuits or spiders/diagrams or qiskit circuits or qiskit DAGs
        """
        
        if (isinstance(obj,qiskit.circuit.quantumcircuit.QuantumCircuit) or isinstance(obj,qiskit.dagcircuit.dagcircuit.DAGCircuit)):
            print('to be added soon')
        
        if (isinstance(obj,zx.graph.graph_s.GraphS) or isinstance(obj,zx.circuit.Circuit)):
            dag=zx_to_dag(g)

        return dag