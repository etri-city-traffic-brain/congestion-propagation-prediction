import numpy as np
import scipy.sparse as sp
# from scipy.sparse import csr_matrix
# from scipy.sparse.csgraph import connected_components

inf = np.infty

class smx_aorm_iterator:
    def __init__(self, A, k=2, shortest_only=False, method = 'sp_mm', cycle_check = False, sparseformat=sp.csr_matrix):
        self.power, self.maxpower = 1, k
        self.n_nodes = A.shape[0]
        self.n_edges = A.nnz
        self.R = sparseformat(A)
        self.Rk = sparseformat(A)
        self.temp = sparseformat(A)
        if cycle_check: self.Rhat = sparseformat(A)
        self.F = sparseformat(A)
        self.F.setdiag(1)
        self.F.data = np.heaviside(self.F.data, 0)
        self.shortest_only = shortest_only
        self.method = method
        self.cycle_check = cycle_check

    def __iter__(self):
        return self

    def __next__(self):
        if self.maxpower > 0 and self.power >= self.maxpower:
            raise StopIteration
        else:
            self.power += 1
            if self.method == 'sp_mm':
              # sparse matrix multiplication
              self.temp = self.R.dot(self.Rk)
              if self.cycle_check:
                  self.Rhat_temp = self.R.dot(self.Rhat)
            else:
              print('Invalid Rk Iterator Method: {}'.format(self.method))
              raise StopIteration

            if self.shortest_only:
              self.temp.data = np.heaviside(self.temp.data, 0)
              diff = self.temp - self.F
              diff.data = np.heaviside(diff.data, 0)
              self.temp = diff
              self.F = self.temp + self.F
              self.F.eliminate_zeros()             

            if self.cycle_check:
              self.Rhat = self.Rhat_temp
              self.Rhat.setdiag(0)
              self.Rhat.data = np.heaviside(self.Rhat.data, 0)
              self.Rhat.eliminate_zeros()

            if self.temp.sum() < 0.5 :
              raise StopIteration

            self.Rk = self.temp
            self.Rk.eliminate_zeros()

        if self.cycle_check:
            return self.Rhat, self.power
        else:
            return self.Rk, self.power

class AormIterator:
    def __init__(self, A, k=2, shortest_only=False, method = 'edge', cycle_check = False):
    # produce R^k
    #  - shortest_only flag : produces shortest path only
    #  - k : step parameter
    #         produces R^1, R^2, R^3, ..., R^k
    #         if k is negative ==> produces R^k until it converges to 0 matrix
        self.power, self.maxpower = 1, k
        self.n_nodes = len(A)
        self.R, self.Rk, self.temp = A.copy(), A.copy(), A.copy()
        if cycle_check: self.Rhat = A.copy()
        self.V = np.heaviside(np.eye(self.n_nodes) + A, 0)
        self.neigh = [np.nonzero(self.R[node, :])[0] for node in range(self.n_nodes)]
        self.shortest_only = shortest_only
        self.method = method
        self.cycle_check = cycle_check

    def __iter__(self):
        return self

    def __next__(self):
        if self.maxpower > 0 and self.power >= self.maxpower:
            raise StopIteration
        else:
            self.power += 1
            if self.method == 'matmult':
              # SIMD powered matrix multiplication
              self.temp = self.R.dot(self.Rk)
              if self.cycle_check:
                  self.Rhat_temp = self.R.dot(self.Rhat)
            elif self.method == 'edge':
              # Edge-wise matrix multiplication
              self.temp = np.zeros((self.n_nodes, self.n_nodes))
              for node in range(self.n_nodes):
                  self.temp[node, :] = (self.Rk[self.neigh[node], :]).sum(axis=0)
              if self.cycle_check:
                  self.Rhat_temp = np.zeros((self.n_nodes, self.n_nodes))
                  for node in range(self.n_nodes):
                      self.Rhat_temp[node, :] = (self.Rhat[self.neigh[node], :]).sum(axis=0)
            else:
              print('Invalid Rk Iterator Method: {}'.format(self.method))
              raise StopIteration

            if self.shortest_only:
                    self.temp = np.heaviside(np.heaviside(self.temp, 0) - self.V, 0)
                    self.V = self.temp + self.V

            if self.cycle_check:
                self.Rhat = self.Rhat_temp.copy()
                np.fill_diagonal(self.Rhat, 0.0)
                self.Rhat= np.heaviside(self.Rhat, 0.0)
            if self.temp.sum() < 0.5 :
                raise StopIteration
            self.Rk = self.temp.copy()
        if self.cycle_check:
            return self.Rhat, self.power
        else:
            return self.Rk, self.power

class AkIterator:
    def __init__(self, A, k=2, method='adjacency', shortest_only=False):
    # produce A^k or R^k based on method
    #  - method : 'adjacency' --- powers of adjacenty matrix
    #             'possibility' --- powers of possibility matrix
    #  - shortest_only flag : produces shortest path only
    #  - k : step parameter
    #         produces A^1, A^2, A^3, ..., A^k
    #         if k is negative ==> produces A^k until it converges to 0 matrix
        self.power, self.maxpower = 1, k
        self.n_nodes = len(A)
        self.A, self.Ak, self.temp = A.copy(), A.copy(), A.copy()
        self.neigh = [np.nonzero(self.A[node, :])[0] for node in range(self.n_nodes)]
        self.visited = [ list(np.nonzero(self.A[node, :])[0]) + [node] for node in range(self.n_nodes)] #[ [i] for i in range(self.n_nodes)]
        self.method = method
        self.shortest_only = shortest_only

    def __iter__(self):
        return self

    def __next__(self):
        if self.maxpower > 0 and self.power >= self.maxpower:
            raise StopIteration
        else:
            self.power += 1
            self.temp = np.zeros((self.n_nodes, self.n_nodes))
            for node in range(self.n_nodes):
                self.temp[node, :] = (self.Ak[self.neigh[node], :]).sum(axis=0)
                if self.shortest_only:
                    self.temp[node, self.visited[node]] = 0
                    self.visited[node] = self.visited[node] + list (np.nonzero(self.temp[node, :])[0])

            if self.method == 'possibility' :
                self.temp = np.heaviside(self.temp, 0.0)
            if self.temp.sum() < 0.5 :
                raise StopIteration
            self.Ak = self.temp.copy()
        return self.Ak, self.power

def Ck_accumAk(A, k):
    n, e = len(A), np.count_nonzero(A)
    mu_degree = e / n
    print('{0} vertex are linked with {1} edges, and mean degree of each node is {2}'.format(n, e, mu_degree))

    np.fill_diagonal(A, 0)
    Ck = A.copy()
    accumBk = A.copy()
    Ak = A.copy()

    for Aks, power in AkIterator(A, k, method='possibility', shortest_only = True):
      Ak_inf = Aks.copy()
      Ak_inf[Ak_inf > 0.5 ] = power
      Ak_inf[Ak_inf < 0.5 ] = np.Infinity
      # Semin operation: selection of element-wise minimum
      Ck = Ck + power * Aks 
      accumBk += Aks / power
    return Ck, accumBk
