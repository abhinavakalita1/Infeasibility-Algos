import numpy as np
import manifold_tracing as mt

# Create a 2D FK triangulation
fk = mt.FKTriangulation(2)          # or CTriangulation(2) for Coxeter A lattice

# Define an SDF (e.g., unit circle)
def sdf(p): return np.linalg.norm(p) - 1.0

seed = np.array([1.0, 0.05])

# Trace the manifold — returns list of intersection points
points = mt.trace_manifold(fk, sdf, seed)

# Locate which simplex contains a point
s = mt.locate_simplex(fk, seed)

# Get faces/cofaces
edges  = mt.faces(s, 1)             # 1-faces (edges) of simplex s
tris   = mt.cofaces(edges[0], 2)   # 2-cofaces of an edge

# Check a single edge
pt = mt.edge_intersection(edges[0], fk, sdf)  # returns np.ndarray or None