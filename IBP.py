import numpy as np
from tqdm import tqdm,trange
import random
import sys
import sympy 
import time

class IBP:
    def __init__(self, X, Z = None, sigma_X = 1, sigma_A = 1, alpha = 1):
        self.X = X
        self.N, self.D = X.shape
        self.sigma_X = sigma_X
        self.sigma_A = sigma_A
        self.trX = np.trace(self.X.T @ self.X)
        self.time_Z = [0, 0]
        self.time_K = [0, 0]
        self.time_alpha = [0,0]
        self.time_sigma_X = [0,0]
        self.time_sigma_A = [0,0]
        self.time_simplify = [0,0]
        self.time_total = 0
        if type(alpha) is tuple:
            self.alpha, self.alpha_a, self.alpha_b = alpha
            self.alpha_update = True
        else:
            self.alpha = alpha
            self.alpha_update = False
        if type(sigma_X) is tuple:
            self.sigma_X, self.sigma_X_a, self.sigma_X_b = sigma_X
            self.sigma_X_update = True
        else:
            self.sigma_X = sigma_X
            self.sigma_X_update = False
        if type(sigma_A) is tuple:
            self.sigma_A, self.sigma_A_a, self.sigma_A_b = sigma_A
            self.sigma_A_update = True
        else:
            self.sigma_A = sigma_A
            self.sigma_A_update = False
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
        _t = time.time()
        history = {
            'Z': [None] * maxiter, 
            'K': np.zeros(maxiter),
            'sigma_X': np.zeros(maxiter),
            'sigma_A': np.zeros(maxiter),
            'alpha': np.zeros(maxiter)
        }
        with tqdm(total=maxiter) as pbar:
            for it in range(maxiter):
                self.step()
                history['Z'][it] = self.Z * 1
                history['K'][it] = self.K
                history['sigma_X'][it] = self.sigma_X
                history['sigma_A'][it] = self.sigma_A
                history['alpha'][it] = self.alpha
                pbar.set_description("Current K = %s" % self.K)
                pbar.update(1)
        self.time_total = time.time() - _t
        return history
    
    def step(self):
        for i in random.sample(range(self.N), self.N):
            self.sampleZ(i)
            _t = time.time()
            self.sampleK(i)
            self.time_K[0] += time.time() - _t
            self.time_K[1] += 1
        _t = time.time()
        # self.simplify()
        self.time_simplify[0] += time.time() - _t
        self.time_simplify[1] += 1
        _t = time.time()
        if self.alpha_update:
            self.sampleAlpha()
        self.time_alpha[0] += time.time() - _t
        self.time_alpha[1] += 1
        _t = time.time()
        if self.sigma_X_update:
            self.sampleSigmaX()
        self.time_sigma_X[0] += time.time() - _t
        self.time_sigma_X[1] += 1
        _t = time.time()
        if self.sigma_A_update:
            self.sampleSigmaA()
        self.time_sigma_A[0] += time.time() - _t
        self.time_sigma_A[1] += 1
    
    def simplify(self):
        _, inds = sympy.Matrix(self.Z).rref()
        self.Z = self.Z[:, inds]
        self.K = self.Z.shape[1]
        return self

    def _lp_original(self, Z = None, sigma_X = None, sigma_A = None):
        if Z is None:
            Z = self.Z
        if sigma_X is None:
            sigma_X = self.sigma_X
        if sigma_A is None:
            sigma_A = self.sigma_A
        K = Z.shape[1]
        invMat = np.linalg.inv(Z.T @ Z + sigma_X**2 / sigma_A**2 * np.eye(K))
        res = - self.N * self.D / 2 * np.log(2 * np.pi)
        res -= (self.N - K) * self.D * np.log(sigma_X)
        res -= K * self.D * np.log(sigma_A)
        res += self.D / 2 * np.log(np.linalg.det(invMat))
        res -= 1 / (2 * sigma_X**2) * np.trace(self.X.T @ (np.eye(self.N) - Z @ invMat @ Z.T) @ self.X)
        return res
        
    def lp(self, Z = None, sigma_X = None, sigma_A = None):
        if Z is None:
            Z = self.Z
        if sigma_X is None:
            sigma_X = self.sigma_X
        if sigma_A is None:
            sigma_A = self.sigma_A
        K = Z.shape[1]
        u, s, _ = np.linalg.svd(Z,full_matrices=False)
        det = np.sum(np.log(s**2 + sigma_X**2 / sigma_A**2))
        l = s**2 / (s**2 + sigma_X**2 / sigma_A**2)
        uTX = u.T @ self.X
        uX = np.sum(uTX ** 2, axis = 1)

        res = - self.N * self.D / 2 * np.log(2 * np.pi)
        res -= (self.N - K) * self.D * np.log(sigma_X)
        res -= K * self.D * np.log(sigma_A)
        res -= self.D / 2 * det
        res -= 1 / (2 * sigma_X**2) * (self.trX - sum(l * uX))
        return res

    def sampleZ(self, i):
        for k in range(self.K):
            # timing
            _t = time.time()
            self._sampleZ(i, k)
            self.time_Z[0] += time.time() - _t
            self.time_Z[1] += 1
    def _sampleZ(self, i, k):
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
        log_prob = np.array([0])
        lmd = self.alpha / self.N
        e = np.zeros(self.Z.shape[0])
        e[i] = 1
        U, S, _ = np.linalg.svd(self.Z, full_matrices = False)
        d = S ** 2 / (S ** 2 + self.sigma_X**2/self.sigma_A**2)
        gamma_i = U @ (d * U[i,])
        cur_diff = cnt = 0
        while cnt + self.K < self.N and cur_diff > log_thres:
            cnt += 1
            mu = 1 + self.sigma_X**2/self.sigma_A**2 - gamma_i[i]
            t = 1/mu * np.sum((self.X.T @ gamma_i - self.X[i])**2)
            cur_diff += self.D * np.log(self.sigma_X / self.sigma_A)
            cur_diff -= self.D / 2 * np.log(mu) 
            cur_diff += t / (2 * self.sigma_X**2)
            cur_diff += np.log(lmd) - np.log(cnt)
            gamma_i += 1/mu * (gamma_i[i] - 1) * (gamma_i - e)
            log_prob = np.append(log_prob, cur_diff)
        
        # avoid overflow/underflow error
        log_prob -= np.max(log_prob)
        prob = np.exp(log_prob)
        prob /= np.sum(prob)

        ### sample using log_prob
        K_post = np.argmax(np.random.multinomial(1, prob))
        m = np.sum(self.Z, axis = 0) - self.Z[i, :]
        self.Z = IBP.append(self.Z[:, m != 0], i, K_post)
        self.K = self.Z.shape[1]
        
    def _sampleK_original(self, i, log_thres = -16):
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
        self.alpha = np.random.gamma(self.alpha_a + self.K, self.alpha_b + np.sum(1/np.arange(1, self.N + 1)))

    def sampleSigmaX(self, epsilon = 0.01):
        new_sigma_X = IBP.wallRandomWalk(self.sigma_X, epsilon, wall = (0, None))
        log_p = (-self.sigma_X_a - 1) * (np.log(new_sigma_X) - np.log(self.sigma_X))
        log_p -= self.sigma_X_b * (1/new_sigma_X - 1/self.sigma_X)
        log_p += self.lp(sigma_X = new_sigma_X) - self.lp()
        if IBP.binary(min(0,log_p), type = 'log'):
            self.sigma_X = new_sigma_X
    
    def sampleSigmaA(self, epsilon = 0.01):
        new_sigma_A = IBP.wallRandomWalk(self.sigma_A, epsilon, wall = (0, None))
        log_p = (-self.sigma_A_a - 1) * (np.log(new_sigma_A) - np.log(self.sigma_A))
        log_p -= self.sigma_A_b * (1/new_sigma_A - 1/self.sigma_A)
        log_p += self.lp(sigma_A = new_sigma_A) - self.lp()
        if IBP.binary(min(0,log_p), type = 'log'):
            self.sigma_A = new_sigma_A

    def postMean(self):
        return np.linalg.inv(self.Z.T @ self.Z + self.sigma_X**2 / self.sigma_A**2 * np.eye(self.K)) @ self.Z.T @ self.X

    def profile(self):
        print(f"Total time used: {self.time_total}s")
        print(f"Update Z\ttot_time {self.time_Z[0]}s\t#exec {self.time_Z[1]}\tavg_time {self.time_Z[0]/self.time_Z[1]}s")
        print(f"Update K\ttot_time {self.time_K[0]}s\t#exec {self.time_K[1]}\tavg_time {self.time_K[0]/self.time_K[1]}s")
        print(f"Update alpha\ttot_time {self.time_alpha[0]}s\t#exec {self.time_alpha[1]}\tavg_time {self.time_alpha[0]/self.time_alpha[1]}s")
        print(f"Update sigma_X\ttot_time {self.time_sigma_X[0]}s\t#exec {self.time_sigma_X[1]}\tavg_time {self.time_sigma_X[0]/self.time_sigma_X[1]}s")
        print(f"Update sigma_A\ttot_time {self.time_sigma_A[0]}s\t#exec {self.time_sigma_A[1]}\tavg_time {self.time_sigma_A[0]/self.time_sigma_A[1]}s")
        print(f"Other\t\ttot_time {self.time_total - self.time_Z[0] - self.time_K[0] - self.time_alpha[0] - self.time_sigma_X[0] - self.time_sigma_A[0]}s")
        return [self.time_Z, self.time_K, self.time_alpha, self.time_sigma_X, self.time_sigma_A, self.time_total]

    @staticmethod
    def binary(p, type = None):
        if type == 'log':
            if p > 0:
                return 1
            elif p < -200:
                return 0
            p = np.exp(p)
        elif type == 'logdiff':
            if p > 200:
                return 1
            elif p < -200:
                return 0
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
    
    @staticmethod
    def wallRandomWalk(X, eps, wall = (None, None)):
        new_X = np.random.normal(X, eps)
        left, right = wall
        def walled(_):
            if left is not None and _ < left:
                return walled(2 * left - _)
            if right is not None and _ > right:
                return walled(2 * right - _)
            return _
        return walled(new_X)


