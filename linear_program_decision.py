import numpy as np 
import matplotlib.pyplot as plt
import cvxpy as cp

import suppressedBD as bd

# relevant functions

class birthdeathSMDP:

	# infinite horizon, average reward MDP
	suppression_cost = 1
	suppression_exponent = 1

	# time-homogenous BD rates

	def __init__(self,n_states,gamma_values,beta,delta=1):
		self.b = beta 
		self.d = delta
		self.J = n_states + 1 
		self.grange = gamma_values

		self.sup_cost = suppression_cost
		self.sup_power = suppression_exponent

	def L(i,g): # aggregate birth rate
		return b*i

	def U(i,g): # aggregate death rate
		if i == 0:
			return 0
		else:
			return d*i + g

	def p(i,g):
		return L(i,g)/(L(i,g) + U(i,g))

	def q(i,g):
		return U(i,g)/(L(i,g) + U(i,g))

	def tb(i,g):
		if i == 0:
			return 0
		else:
			return 1/(L(i,g) + U(i,g))

	def rb(i,g):
		return -1*(L(i,g) - U(i,g))/(L(i,g)+U(i,g)) - self.sup_cost*g**(self.sup_power)



	def setUpProblem(self,n_states):

		


