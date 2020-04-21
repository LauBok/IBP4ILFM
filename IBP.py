import numpy as np

class IBP:
    def __init__(self, X, Z = None, sigma_X = 1, sigma_A = 1, alpha = 1):
        self.X = X
        self.N, self.D = X.shape
        self.sigma_X = sigma_X
        self.sigma_A = sigma_A
        self.alpha = alpha
        if Z is None:
            self.initZ()
        else:
            assert(self.N == Z.shape[0])
            self.Z = Z
            self.K = Z.shape[1]

    def initZ(self):
        ## initialize self.Z and update self.K
        pass
    
    def MCMC(self, maxiter = 1000):
        history = {
            'Z': [None] * maxiter, 
            'K': np.zeros(maxiter),
            'sigma_X': np.zeros(maxiter),
            'sigma_A': np.zeros(maxiter),
            'alpha': np.zeros(maxiter)
        }
        for it in range(maxiter):
            self.step()
            history['Z'][it] = self.Z * 1
            history['K'][it] = self.K
            history['sigma_X'][it] = self.sigma_X
            history['sigma_A'][it] = self.sigma_A
            history['alpha'][it] = self.alpha
        return history
    
    def step(self):
        for i in range(self.N):
            for k in range(self.K):
                self.sampleZ(i, k)
            self.sampleK(i)
        self.sampleAlpha()

    def sampleZ(self, i, k):
        # Use formula (9) to update Z_{ik}
        # delete pass when you implement this
        pass
    
    def sampleK(self, i):
        pass
    
    def sampleAlpha(self):
        pass

    def postMean(self):
        return np.linalg.inv(self.Z.T @ self.Z + self.sigma_X**2 / self.sigma_A**2 * np.eye(self.D)) @ self.Z.T @ self.X

    @staticmethod
    def binary(p, type = None):
        if type is None:
            assert(0 <= p <= 1)
        elif type is 'log':
            assert(p <= 0)
            p = np.exp(p)
        elif type is 'logdiff':
            p = np.exp(p) / (1 + np.exp(p))
        return np.random.random() < p
    