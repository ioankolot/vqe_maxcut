import math
from quantum_circuit import VQE
import networkx as nx
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt




number_of_qubits = 4
graph = nx.gnm_random_graph(number_of_qubits, m=np.random.randint(2*number_of_qubits, 3*number_of_qubits), seed=10) #we use a random graph
w = nx.to_numpy_matrix(graph, nodelist=sorted(graph.nodes()))


#Choose layers for the Hardware Efficient Ansatz.
layers = 1

#Choose initial angles at random.
thetas = [np.random.uniform(0, 2*np.pi) for _ in range((layers+1)*number_of_qubits)]

#Intiliaze and visualize the quantum circuit

vqe_circuit = VQE(number_of_qubits, w, graph, thetas, layers)
vqe_circuit.vqe.draw(output='mpl')
plt.show()

#You can calculate the optimal cost value by brute force.
print(f'The optimal cost value is {vqe_circuit.best_cost_brute()}')


def expectation_value(x): #this function return the expectation value as function of the parameters and will be used for the classical optimization part
    vqe_circuit = VQE(number_of_qubits, w, graph, x, layers)
    exp_value = vqe_circuit.get_expected_value()
    return exp_value


opt = scipy.optimize.minimize(expectation_value, x0=tuple(thetas), method='COBYLA')

print('\n')
print(f'The optimal expectation value found by VQE is {-opt.fun}')

