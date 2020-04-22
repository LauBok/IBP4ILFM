import numpy as np
from tqdm import tqdm

class IBP:
    def __init__(self, X, Z = None, sigma_X = 1, sigma_A = 1, alpha = 1):
        self.X = X
        self.N, self.D = X.shape
        self.sigma_X = sigma_X
        self.sigma_A = sigma_A
        if type(alpha) is tuple:
            self.alpha_a, self.alpha_b = alpha
            self.alpha = np.random.gamma(*alpha)
            self.alpha_update = True
        else:
            self.alpha = alpha
            self.alpha_update = False
        if Z is None:
            self.initZ()
        else:
            assert(self.N == Z.shape[0])
            self.Z = Z
            self.K = Z.shape[1]

    def initZ(self):
        Z = np.zeros((self.N, 0))
        K = []
        for i in range(self.N):
            # the i-th customer
            for k, nk in enumerate(K):
                Z[i, k] = IBP.binary(nk / (i + 1))
            K_new = np.random.poisson(self.alpha / (i + 1))
            K += [1] * K_new
            Z = IBP.append(Z, i, K_new)
        ## initialize self.Z and update self.K
        self.Z = Z
        self.K = len(K)
        assert(len(K) == Z.shape[1])
    
    def MCMC(self, maxiter = 1000):
        history = {
            'Z': [None] * maxiter, 
            'K': np.zeros(maxiter),
            'sigma_X': np.zeros(maxiter),
            'sigma_A': np.zeros(maxiter),
            'alpha': np.zeros(maxiter)
        }
        for it in tqdm(range(maxiter)):
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

    def lp(self, Z):
        K = Z.shape[1]
        invMat = np.linalg.inv(Z.T @ Z + self.sigma_X**2 / self.sigma_A**2 * np.eye(K))
        res = - self.N * self.D / 2 * np.log(2 * np.pi)
        res -= (self.N - K) * self.D * np.log(self.sigma_X)
        res -= K * self.D * np.log(self.sigma_A)
        res += self.D / 2 * np.log(np.linalg.det(invMat))
        res -= 1 / (2 * self.sigma_X**2) * np.trace(self.X.T @ (np.eye(self.N) - Z @ invMat @ Z.T) @ self.X)
        return res

    def sampleZ(self, i, k):
        # Use formula (9) to update Z_{ik}
        mk = sum(self.Z[:, k]) - self.Z[i, k]
        ## The column only has this non-zero entry.
        if mk == 0:
            self.Z[i, k] = 0
        else:
            Z0 = IBP.copy(self.Z)
            Z0[i, k] = 0
            Z1 = IBP.copy(self.Z)
            Z1[i, k] = 1
            logpratio = self.lp(Z1) - self.lp(Z0) + np.log(mk) - np.log(self.N - mk)
            self.Z[i, k] = IBP.binary(logpratio, "logdiff")
    
    def sampleK(self, i, log_thres = -16):
        log_prob = np.array([])
        K_new = 0
        lmd = self.alpha / self.N
        log_fac = lambda n: 0 if n == 0 else np.log(n) + log_fac(n - 1)
        log_pois = lambda K: K * np.log(lmd) - lmd - log_fac(K)
       
        while log_pois(K_new) > log_thres:
            log_prob = np.append(log_prob, log_pois(K_new) + self.lp(IBP.append(self.Z, i, K_new)))
            K_new += 1

        # avoid overflow/underflow error
        log_prob -= np.max(log_prob)
        prob = np.exp(log_prob)
        prob /= np.sum(prob)

        ### sample using log_prob
        K_post = np.argmax(np.random.multinomial(1, prob))
        m = np.sum(self.Z, axis = 0) - self.Z[i, :]
        self.Z = IBP.append(self.Z[:, m != 0], i, K_post)
        self.K = self.Z.shape[1]
    
    def sampleAlpha(self):
        if self.alpha_update:
            self.alpha = np.random.gamma(self.alpha_a + self.K, self.alpha_b + np.sum(1/np.arange(1, self.N + 1)))

    def postMean(self):
        return np.linalg.inv(self.Z.T @ self.Z + self.sigma_X**2 / self.sigma_A**2 * np.eye(self.K)) @ self.Z.T @ self.X

    @staticmethod
    def binary(p, type = None):
        if type is None:
            assert(0 <= p <= 1)
        elif type == 'log':
            assert(p <= 0)
            p = np.exp(p)
        elif type == 'logdiff':
            p = np.exp(p) / (1 + np.exp(p))
        return np.random.random() < p
    
    @staticmethod
    def copy(mat):
        return mat + 0

    @staticmethod
    def append(Z, i, K_new):
        N, K = Z.shape
        _Z = np.zeros((N, K + K_new))
        _Z[:, :K] = Z
        _Z[i, K:] = 1
        return _Z


