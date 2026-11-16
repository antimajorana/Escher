"""Graph traversal helpers shared by defect construction and classification."""
from collections import deque


def layer_signature(adj, deg, v, r) -> tuple:
    """Multiset of (depth, degree, #nbrs one layer in, #same layer, #one layer out) over the ball."""
    dist = {v: 0}
    frontier = [v]
    for k in range(1, r + 1):
        nxt = []
        for u in frontier:
            for w in adj[u]:
                if w not in dist:
                    dist[w] = k
                    nxt.append(w)
        frontier = nxt
    sig = []
    for u, du in dist.items():
        a = b = c = 0
        for w in adj[u]:
            dw = dist.get(w)
            if dw is None:
                continue
            if dw == du - 1:
                a += 1
            elif dw == du:
                b += 1
            else:
                c += 1
        sig.append((du, deg[u], a, b, c))
    return tuple(sorted(sig))


def _multi_bfs(adj, sources, cutoff=None) -> dict:
    dist = {s: 0 for s in sources}
    dq = deque(sources)
    while dq:
        u = dq.popleft()
        if cutoff is not None and dist[u] >= cutoff:
            continue
        for w in adj[u]:
            if w not in dist:
                dist[w] = dist[u] + 1
                dq.append(w)
    return dist


def _orbits(items, ops, image, same):
    """Orbits of items under ops (union-find). Returns lists of indices, in order of first member."""
    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i, it in enumerate(items):
        for g in ops:
            im = image(it, g)
            for j, other in enumerate(items):
                if same(im, other):
                    a, b = find(i), find(j)
                    if a != b:
                        parent[max(a, b)] = min(a, b)
                    break
    out = {}
    for i in range(len(items)):
        out.setdefault(find(i), []).append(i)
    return list(out.values())
