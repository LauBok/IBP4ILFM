import numpy as np
from IBP import IBP
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

V1 = np.array([
    1, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0
])
V2 = np.array([
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0,
    1, 1, 1, 0, 0, 0
])
V3 = np.array([
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 1,
    0, 0, 0, 0, 1, 1
])
V4 = np.array([
    0, 0, 0, 0, 1, 1,
    0, 0, 0, 0, 0, 1,
    0, 0, 0, 0, 0, 1,
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0
])
V5 = np.array([
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0,
    0, 1, 1, 1, 0, 0,
    0, 0, 1, 0, 0, 0,
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0
])
V = [V1, V2, V3, V4, V5]

N = 200
D1 = 0.6 * np.outer(np.random.binomial(1, 0.5, N), V1)
D2 = 0.6 * np.outer(np.random.binomial(1, 0.5, N), V2)
D3 = 0.6 * np.outer(np.random.binomial(1, 0.5, N), V3)
D4 = 0.6 * np.outer(np.random.binomial(1, 0.5, N), V4)
D5 = 0.6 * np.outer(np.random.binomial(1, 0.5, N), V5)
X = D1 + D2 + D3 + D4 + D5 + 0.2
X += np.random.normal(0, 0.1, X.shape)
print(X.shape)

ibp = IBP(X, alpha = (1,1), sigma_X = (1,1), sigma_A = (1,1))
history = ibp.MCMC(500)
print("K", history["K"])
print("alpha", history["alpha"])
print("sigma_X", history["sigma_X"])
print("sigma_A", history["sigma_A"])

A = ibp.postMean()
f = plt.figure(figsize = (8,8))
gs0 = gridspec.GridSpec(1, 2, figure=f)
gs00 = gridspec.GridSpecFromSubplotSpec(5, 1, subplot_spec=gs0[0])
for i in range(5):
    f.add_subplot(gs00[i, 0]).imshow(V[i].reshape(6, 6))
gs01 = gridspec.GridSpecFromSubplotSpec(5, (A.shape[0] - 1) // 5 + 1, subplot_spec=gs0[1])
for i in range(A.shape[0]):
    f.add_subplot(gs01[i % 5, i // 5]).imshow(A[i].reshape(6, 6))
plt.show()