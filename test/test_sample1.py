import numpy as np
from IBP import IBP
from matplotlib import pyplot as plt

N = 200
V1 = np.random.rand(4)
V2 = np.random.rand(3)
V3 = np.random.rand(8)
V4 = np.random.rand(6)
D1 = np.outer(np.random.binomial(1, 0.5, N), V1)
D2 = np.outer(np.random.binomial(1, 0.5, N), V2)
D3 = np.outer(np.random.binomial(1, 0.5, N), V3)
D4 = np.outer(np.random.binomial(1, 0.5, N), V4)
X = np.c_[D1,D2,D3,D4]
print(X.shape)
#X += np.random.normal(0, 0.1, X.shape)
fig, ax = plt.subplots(1, 3, figsize = (8,6))
ax[0].imshow(X)
ibp = IBP(X, alpha = (1,1), sigma_X = 0.1)
history = ibp.MCMC(100)
print(history["K"])
print(history["alpha"])
ax[1].imshow(ibp.Z @ ibp.postMean())
ax[2].imshow(X - ibp.Z @ ibp.postMean())
plt.show()