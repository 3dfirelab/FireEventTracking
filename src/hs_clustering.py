import numpy as np
from collections import defaultdict
from joblib import Parallel, delayed, cpu_count
import math
import threading
try:
    from numba import njit
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False

    def njit(*args, **kwargs):
        def _wrap(func):
            return func
        return _wrap

class UnionFind:
    def __init__(self, n):
        self.parent = np.arange(n, dtype=np.int32)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        else:
            self.parent[ry] = rx
            if self.rank[rx] == self.rank[ry]:
                self.rank[rx] += 1


@njit(cache=True)
def _uf_find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


@njit(cache=True)
def _uf_union(parent, rank, x, y):
    rx = _uf_find(parent, x)
    ry = _uf_find(parent, y)
    if rx == ry:
        return
    if rank[rx] < rank[ry]:
        parent[rx] = ry
    else:
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1


@njit(cache=True)
def _numba_cluster(X, idx_sorted, start_idx, counts, unique_keys, base, cell_cx, cell_cy, eps2):
    n = X.shape[0]
    parent = np.arange(n, dtype=np.int64)
    rank = np.zeros(n, dtype=np.int8)
    n_cells = unique_keys.shape[0]
    for ci in range(n_cells):
        key = unique_keys[ci]
        start_i = start_idx[ci]
        end_i = start_i + counts[ci]
        cx = cell_cx[ci]
        cy = cell_cy[ci]
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                ncx = cx + dx
                ncy = cy + dy
                if ncx < 0 or ncy < 0:
                    continue
                neigh_key = ncx * base + ncy
                if neigh_key < key:
                    continue
                j = np.searchsorted(unique_keys, neigh_key)
                if j >= n_cells or unique_keys[j] != neigh_key:
                    continue
                start_j = start_idx[j]
                end_j = start_j + counts[j]
                for ii in range(start_i, end_i):
                    pi = X[idx_sorted[ii]]
                    jj_start = ii + 1 if j == ci else start_j
                    for jj in range(jj_start, end_j):
                        pj = X[idx_sorted[jj]]
                        dxv = pj[0] - pi[0]
                        dyv = pj[1] - pi[1]
                        if dxv * dxv + dyv * dyv <= eps2:
                            _uf_union(parent, rank, idx_sorted[ii], idx_sorted[jj])
    for i in range(n):
        parent[i] = _uf_find(parent, i)
    return parent


class ParallelEpsDBSCAN:
    """
    Drop-in replacement for DBSCAN(eps=..., min_samples=1, metric='euclidean')
    using parallel ε-connectivity (graph connected components).
    """

    def __init__(self, eps, n_jobs=-1, use_numba=True):
        self.eps = eps
        self.n_jobs = n_jobs
        self.use_numba = use_numba

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]

        # grid size = eps / 2 for density robustness
        h = self.eps / 2.0
        inv_h = 1.0 / h

        # spatial binning
        cx = np.floor(X[:, 0] * inv_h).astype(np.int32)
        cy = np.floor(X[:, 1] * inv_h).astype(np.int32)

        eps2 = self.eps * self.eps

        if self.use_numba and _HAS_NUMBA:
            min_cx = int(cx.min())
            min_cy = int(cy.min())
            cx_off = (cx - min_cx).astype(np.int64)
            cy_off = (cy - min_cy).astype(np.int64)
            base = int(cy_off.max()) + 1
            cell_key = cx_off * base + cy_off

            order = np.argsort(cell_key, kind="mergesort")
            keys_sorted = cell_key[order]
            unique_keys, start_idx, counts = np.unique(
                keys_sorted, return_index=True, return_counts=True
            )
            cell_cx = (unique_keys // base).astype(np.int64)
            cell_cy = (unique_keys - cell_cx * base).astype(np.int64)

            parent = _numba_cluster(
                X,
                order.astype(np.int64),
                start_idx.astype(np.int64),
                counts.astype(np.int64),
                unique_keys.astype(np.int64),
                np.int64(base),
                cell_cx,
                cell_cy,
                eps2,
            )
            _, labels = np.unique(parent, return_inverse=True)
            self.labels_ = labels
            return self

        cells = defaultdict(list)
        for i in range(n):
            cells[(cx[i], cy[i])].append(i)

        cell_keys = list(cells.keys())
        cell_keys.sort()

        # 7x7 neighborhood (because h = eps / 2 allows cell deltas up to 3)
        offsets = [(dx, dy) for dx in range(-3, 4) for dy in range(-3, 4)]

        uf = UnionFind(n)
        uf_lock = threading.Lock()

        def process_block(block_cells):
            for cell in block_cells:
                cx, cy = cell
                idx_i = cells[cell]
                Xi = X[idx_i]

                for dx, dy in offsets:
                    neigh = (cx + dx, cy + dy)
                    if neigh not in cells or neigh < cell:
                        continue

                    idx_j = cells[neigh]
                    Xj = X[idx_j]

                    for ii, pi in enumerate(Xi):
                        jj_start = ii + 1 if neigh == cell else 0
                        diff = Xj[jj_start:] - pi
                        dist2 = diff[:, 0] ** 2 + diff[:, 1] ** 2
                        mask = dist2 <= eps2
                        if not np.any(mask):
                            continue

                        matches = np.where(mask)[0]
                        for k in matches:
                            i_idx = idx_i[ii]
                            j_idx = idx_j[jj_start + k]
                            with uf_lock:
                                uf.union(i_idx, j_idx)

        # parallel block processing over cells
        if self.n_jobs is None:
            eff_jobs = 1
        elif self.n_jobs > 0:
            eff_jobs = self.n_jobs
        else:
            eff_jobs = max(1, cpu_count() + 1 + self.n_jobs)

        block_size = max(1, int(math.ceil(len(cell_keys) / eff_jobs)))
        blocks = [
            cell_keys[i : i + block_size]
            for i in range(0, len(cell_keys), block_size)
        ]
        Parallel(n_jobs=eff_jobs, prefer="threads")(
            delayed(process_block)(block) for block in blocks
        )

        # final labels
        roots = np.array([uf.find(i) for i in range(n)])
        _, labels = np.unique(roots, return_inverse=True)

        self.labels_ = labels
        return self
