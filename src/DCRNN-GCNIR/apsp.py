import numpy as np
from scipy.sparse.csr import csr_matrix
from utils import *
from aorm import *
from tqdm import tqdm

def apsp_sparse_AormIterator(A, k=-1, method = 'sp_mm', sparseformat=csr_matrix):
    A.setdiag(0)
    A.eliminate_zeros()
    n, e = A.shape[0], A.nnz
    # mu_degree = round(2*e / n, 2)
    # print(f'# |V|: {n}, |E|: {e}, mean degree: {mu_degree}')
    
    D = A.copy()
    
    for Rks, power in tqdm(smx_aorm_iterator(A, k, shortest_only=True, method = method, sparseformat=sparseformat), desc='# Sparse AORM'):
      D = D + power * Rks
    return D

def apsp_AormIterator(A, k=-1, method = 'edge'):
    np.fill_diagonal(A, 0)

    n, e = len(A), np.count_nonzero(A)
    mu_degree = round(2*e / n, 2)
    print(f'# |V|: {n}, |E|: {e}, mean degree: {mu_degree}')
    
    D = A.copy()
    Ak = A.copy()
    for Rks, power in tqdm(AormIterator(A, k, shortest_only=True, method = method), desc='# incremental computation of AORM'):
      D = D + power * Rks
    return D

def apsp_bfs_queue(A): # All-pair shortest path with BFS using Queue and adjacent list
    n_nodes = len(A)
    adj_list = [ np.array(np.nonzero(A[node])).flatten() for node in range(n_nodes) ]
    D = A.copy()
    for node in tqdm(range(n_nodes)): # bfs for each node
        step = 1
        Q = [node]
        visited = np.zeros(n_nodes).astype(int); visited[node] = 1
        while( len(Q) > 0 ) :
            last_visited = Q.copy() # retrieve all elements from Queue
            Q = []; new_nodes = []  # initialize queue and new visits
            for i in last_visited :
                new_nodes = [*new_nodes , *adj_list[i]]
            for i in new_nodes:
                if visited[i] == 0:
                    Q.append(i)
                    visited[i] = 1
                    D[node, i] = step
            step += 1
    return D

def apsp_bfs_matrix(A): # All-pair shortest path with BFS using Array Operations on adjacency matrix
    A = A.astype(np.float)
    n_nodes = len(A)
    D = A.copy()
    for node in tqdm(range(n_nodes)): # bfs for each node
        step = 1
        visited = np.zeros(n_nodes); visited[node] = 1
        last_visited = np.zeros(n_nodes); last_visited[node] = 1
        while( last_visited.sum() > 0.5 ) :
            new_nodes = np.array(np.heaviside( A[np.nonzero(last_visited), :].sum(axis=1).flatten(), 0 ))
            last_visited = np.heaviside(new_nodes - visited, 0)
            visited = np.heaviside(visited + last_visited, 0)
            D[node, np.nonzero(last_visited)] = step
            step += 1
    return D, step-1

def apsp_logical_bfs_matrix(A): # All-pair shortest path with BFS using Array Operations on adjacency matrix
    A = A.astype(np.int8)
    n_nodes = len(A)
    D = A.copy()
    for node in pit(range(n_nodes), color = 'magenta'): # bfs for each node
        step = 1
        visited = np.zeros(n_nodes); visited[node] = 1
        last_visited = np.zeros(n_nodes); last_visited[node] = 1
        while( last_visited.sum() > 0.5) :
            new_nodes = np.array(np.heaviside( A[np.nonzero(last_visited), :].sum(axis=1).flatten(), 0 ))
            # new_nodes = np.array(A[np.nonzero(last_visited), :].sum(axis=1).flatten() & ~0 )
            last_visited = np.heaviside(new_nodes - visited, 0)
            # last_visited = np.bitwise_xor(new_nodes.astype(np.bool), visited.astype(np.bool))
            visited = np.bitwise_or(visited.astype(np.bool), last_visited.astype(np.bool))
            D[node, np.nonzero(last_visited)] = step
            step += 1
    return D

def apsp_inc_bfs_matrix(A, cutoff=-1): # All-pair shortest path with BFS using Array Operations on adjacency matrix
    A = A.astype(np.int8)
    n_nodes = len(A)
    D = A.copy()
    for node in pit(range(n_nodes), color = 'magenta'): # bfs for each node
        step = 1
        visited = np.zeros(n_nodes); visited[node] = 1
        last_visited = np.zeros(n_nodes); last_visited[node] = 1
        while( last_visited.sum() > 0.5 and cutoff >= step) :
            # new_nodes = np.array(np.heaviside( A[np.nonzero(last_visited), :].sum(axis=1).flatten(), 0 ))
            new_nodes = np.array(A[np.nonzero(last_visited), :].sum(axis=1).flatten() & ~0 )
            last_visited = np.heaviside(new_nodes - visited, 0)
            # visited = np.heaviside(visited + last_visited, 0)
            visited = np.bitwise_or(visited.astype(np.bool), last_visited.astype(np.bool))
            D[node, np.nonzero(last_visited)] = step
            step += 1
    return D, step-1

def apd(A, n: int):
    """Compute the shortest-paths lengths."""
    if all(A[i][j] for i in range(n) for j in range(n) if i != j):
        return A
    Z = A ** 2
    B = np.matrix([
        [1 if i != j and (A[i][j] == 1 or Z[i][j] > 0) else 0 for j in range(n)]
    for i in range(n)])
    T = apd(B, n)
    X = T*A
    degree = [sum(A[i][j] for j in range(n)) for i in range(n)]
    D = np.matrix([
        [2 * T[i][j] if X[i][j] >= T[i][j] * degree[j] else 2 * T[i][j] - 1 for j in range(n)]
    for i in range(n)])
    return D

def apsp_seidel(A, n):
    """Compute the shortest-paths lengths."""
    # Raimund Seidel's Algorithm proposed in 1995
    if all(A[i,j] for i in pit(range(n), color='white') for j in range(n) if i != j):
        return A
    Z = A.dot(A)
    B = np.array([ [1 if i != j and (A[i,j] > 0 or Z[i,j] > 0) else 0 for j in range(n) ] for i in pit(range(n), color='magenta') ])
    if all(B[i,j] 
    for i in pit(range(n), color = 'green')
    # for i in range(n)
    for j in range(n)if i != j):
        return 2 * B - A
    # T = tqdm(apsp_seidel(B, n), desc='# recursive P-SM')
    T = apsp_seidel(B, n)
    X = T.dot(A)
    degree = [sum(A[i,j] for j in range(n)) for i in pit(range(n), color = 'cyan')]
    D = np.array([[2 * T[i,j] if X[i,j] >= T[i,j] * degree[j] else 2 * T[i,j] - 1 for j in range(n)] for i in pit(range(n), color='gray')])
    return D

def apsp_seidel_tqdm(A, n, bar):
    bar.update(1)
    # sleep(0.01)  # slow-down things a little bit
    """Compute the shortest-paths lengths."""
    # Raimund Seidel's Algorithm proposed in 1995
    if all(A[i,j] for i in range(n) for j in range(n) if i != j):
        return A
    Z = A.dot(A)
    B = np.array([ [1 if i != j and (A[i,j] > 0 or Z[i,j] > 0) else 0 for j in range(n) ] for i in range(n) ])
    if all(B[i,j] for i in range(n) for j in range(n) if i != j):
        return 2 * B - A
    T = apsp_seidel_tqdm(B, n, bar)
    X = T.dot(A)
    degree = [sum(A[i,j] for j in range(n)) for i in range(n)]
    D = np.array([[2 * T[i,j] if X[i,j] >= T[i,j] * degree[j] else 2 * T[i,j] - 1 for j in range(n)] for i in range(n)])
    return D