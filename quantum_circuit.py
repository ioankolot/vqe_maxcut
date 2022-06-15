from qiskit import QuantumCircuit, execute, Aer, QuantumRegister, ClassicalRegister
from qiskit.visualization import *
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt



class VQE():
    def __init__(self, number_of_qubits, w, graph, thetas, layers):

        self.number_of_qubits = number_of_qubits
        self.w = w
        self.graph = graph
        self.init_thetas = thetas[0:self.number_of_qubits]
        self.thetas = thetas
        self.layers = layers
        self.shots = 1000 #You can adjust the accuracy of the calculation of the expectation value.

        self.qreg = QuantumRegister(self.number_of_qubits, name = 'q')
        self.creg = ClassicalRegister(self.number_of_qubits, name = 'c')
        self.vqe = QuantumCircuit(self.qreg, self.creg)

        for qubit in range(self.number_of_qubits): #This is a layer of R_y rotations (zeroth layer)
            self.vqe.ry(self.init_thetas[qubit], qubit)

        self.vqe.barrier()

        for layer in range(layers):  #You can choose the number of layers or change the gates (here we use cz gates and ry rotations for each layer)
            for qubit1 in range(self.number_of_qubits):
                for qubit2 in range(self.number_of_qubits):
                    if qubit1 < qubit2:
                        self.vqe.cz(qubit1, qubit2)

            for qubit in range(self.number_of_qubits):
                self.vqe.ry(self.thetas[(layer+1)*self.number_of_qubits + qubit], qubit)

        self.vqe.barrier()

        self.vqe.measure(range(self.number_of_qubits), self.creg)
        self.counts = execute(self.vqe, Aer.get_backend('qasm_simulator'), shots = self.shots).result().get_counts()

    def get_expected_value(self):
        avr_c = 0
        for sample in list(self.counts.keys()):
            y = [int(num) for num in list(sample)]
            tmp_eng = self.cost_hamiltonian(y) + self.get_offset()
            avr_c += self.counts[sample] * tmp_eng
        energy_expectation = avr_c/self.shots
        return energy_expectation

    def cost_hamiltonian(self, x):
        spins = []
        for i in x[::-1]:
            spins.append(int(i))
        total_energy = 0
        for i in range(self.number_of_qubits):
            for j in range(self.number_of_qubits):
                if i<j:
                    if self.w[i,j] != 0:
                        total_energy += self.w[i,j] * (self.sigma(spins[i]) * self.sigma(spins[j]))
        total_energy /= 2
        return total_energy

    def sigma(self, z):
        if z == 0:
            value = 1
        elif z == 1:
            value = -1
        return value

    def get_offset(self): #the constant part of the Hamiltonian
        offset = 0
        for i in range(self.number_of_qubits):
            for j in range(self.number_of_qubits):
                if i<j:
                    offset -= self.w[i,j]/2
        return offset


    def best_cost_brute(self): #you can use this to classically calculate the optimal expectation value.
        best_cost = 0
        for b in range(2**self.number_of_qubits):
            x = [int(t) for t in reversed(list(bin(b)[2:].zfill(self.number_of_qubits)))]
            cost = 0
            for i in range(self.number_of_qubits):
                for j in range(self.number_of_qubits):
                    cost += self.w[i,j] * x[i] * (1-x[j])
            if best_cost < cost:
                best_cost = cost
        return best_cost

    def probability_of_optimal(self): #this function calculates the overlap with the optimal solution
        optimal_solution = self.best_cost_brute()
        energies = self.exact_counts()
        print(f'The optimal objective value is {-optimal_solution}')
        total_counts_of_optimal = 0
        for energy in energies:
            if round(energy,2) == -round(optimal_solution,2):
                total_counts_of_optimal += 1
        return total_counts_of_optimal/self.shots


    def exact_counts(self):
        energies = []
        for sample in list(self.counts.keys()):
            y = [int(num) for num in list(sample)]
            tmp_eng = self.cost_hamiltonian(y) + self.get_offset()
            for num in range(self.counts[sample]):
                energies.append(tmp_eng)
        energies.sort(reverse=False)
        return energies
