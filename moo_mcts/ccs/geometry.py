import numpy as np

# Numerical tolerance for "equal" coordinates. Real evals are noisy; a small
# epsilon avoids spurious near-duplicate front points.
EPS = 1e-9

def weakly_dominates(u: np.ndarray, v: np.ndarray) -> bool:
    """Return whether u weakly Pareto-dominates v, i.e. u >= v on every axis (CHMCTS Def. 3).

    - u: candidate dominating vector.
    - v: vector being tested for domination.
    """
    return bool(np.all(u >= v - EPS))


def dominates(u: np.ndarray, v: np.ndarray) -> bool:
    """Return whether u Pareto-dominates v, i.e. u >= v everywhere AND u > v somewhere (CHMCTS Def. 3).

    - u: candidate dominating vector.
    - v: vector being tested for domination.
    """
    return bool(np.all(u >= v - EPS) and np.any(u > v + EPS))


def pareto_front(points: np.ndarray) -> np.ndarray:
    """Return the indices of the non-dominated rows of `points` (the Pareto front).

    A row is kept iff no other row dominates it. Exact duplicates (within EPS)
    collapse to a single representative, the one with the lower index.

    - points: an (N, D) array of maximize-form vectors.
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    n = pts.shape[0]
    if n == 0:
        return np.empty(0, dtype=int)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j or not keep[j]:
                continue
            if dominates(pts[j], pts[i]):
                keep[i] = False
                break
            # exact-duplicate tie-break: keep the lower index only
            if np.all(np.abs(pts[i] - pts[j]) <= EPS) and j < i:
                keep[i] = False
                break
    return np.flatnonzero(keep)


def scalarize(points: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Return the linear scalarization w . v for each row of points.

    - points: an (N, D) array of value vectors.
    - w: weight vector dotted against each row.
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    w = np.asarray(w, dtype=float)
    return pts @ w


def convex_hull_vertices(points: np.ndarray) -> np.ndarray:
    """Return the indices of points optimal for some nonnegative weight vector.
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    n, d = pts.shape
    if n <= 1:
        return np.arange(n)
    front = pareto_front(pts)
    fpts = pts[front]
    keep_local: set[int] = set()

    # Axis-aligned directions guarantee the per-objective optima are included.
    dirs = list(np.eye(d))
    # A deterministic spread of simplex directions for the interior of the hull.
    rng = np.random.default_rng(0)
    for _ in range(max(64, 16 * d)):
        g = rng.random(d)
        dirs.append(g / g.sum())

    for w in dirs:
        s = fpts @ np.asarray(w, dtype=float)
        keep_local.add(int(np.argmax(s)))

    return front[sorted(keep_local)]


def optimistic_point(ref_point: np.ndarray, best_seen: np.ndarray | None) -> np.ndarray:
    """Return an optimistic (dominating) vector for cold-start initialization.
    """
    ref_point = np.asarray(ref_point, dtype=float)
    if best_seen is None or len(best_seen) == 0:
        # one "optimism unit" above the reference on each axis
        return ref_point + np.abs(ref_point) + 1.0
    bs = np.atleast_2d(np.asarray(best_seen, dtype=float))
    return bs.max(axis=0)
