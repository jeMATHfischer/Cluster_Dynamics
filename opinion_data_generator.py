import numpy as np 

np.random.seed(42)

for i in range(10):
    n = 10
    X = np.random.uniform(0,1,n)
    np.savetxt('initial_opinions_n_{}_i_{}.txt'.format(n, i), X)
