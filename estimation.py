# Testing parameter estimation

import numpy as np 
import matplotlib.pyplot as plt
 
import suppressedBD as bd

if __name__ == "__main__":
	# Generate synthetic test data
	btest = 1.1
	dtest = 1
	gtest = 1.5
	init = 20

	sim = bd.BDensemble(btest,dtest,gtest,init,1000,1)

	traj = sim.run_trajectory()
	print("ran trajectory.")
	js,Fs,ts = traj

	# discretely observe the synthetic process
	Nobs = 15

	obsInds = []
	for t in np.linspace(0,max(ts),Nobs+2):
	    obsInds.append(np.argmin(np.abs(ts-t)))

	observations = []
	for i in range(Nobs):
	    i1 = obsInds[i]
	    i2 = obsInds[i+1]
	    obs = (js[i1],js[i2],ts[i2] - ts[i1])
	    if obs[2] > 0:
	        observations.append(obs)
	        print(obs)
	    else: 
	        continue
	    
	# print(observations)
	plt.plot(ts,js)
	plt.scatter(ts[obsInds[1:-1]],js[obsInds[1:-1]],color = 'red',label = 'observations')
	plt.legend()
	plt.xlabel(r"time $t$")
	plt.ylabel(r"population $j(t)$")
	plt.title(r"$\beta = {},\ \delta = {},\ \gamma = {}$".format(btest,dtest,gtest))
	plt.show()


	rez = bd.BDestimate(observations)

	print("FINAL ESTIMATE:")
	print(rez)












