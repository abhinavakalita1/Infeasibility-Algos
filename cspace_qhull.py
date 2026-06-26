import pybullet_data
import numpy as np
import pybullet as p
import time, os, glob, itertools, math
import pyvista as pv
from scipy.spatial import ConvexHull
from collections import deque

_T0 = time.perf_counter()
def elapsed(): return f"{time.perf_counter()-_T0:.1f}s"

# ══════════════════════════════════════════════════════════════════
# PYBULLET SETUP
# ══════════════════════════════════════════════════════════════════

physicsClient = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
p.loadURDF("plane.urdf")

arm3Id = p.loadURDF(
    "arm_3.urdf", basePosition=[0, 0, 0],
    baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
    useFixedBase=True,
    flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION)
print(f"[INFO] Arm loaded — {p.getNumJoints(arm3Id)} joints")

hull_body_ids = []
for obj_path in sorted(glob.glob(os.path.join("objs", "*.obj"))):
    try:
        col_id = p.createCollisionShape(p.GEOM_MESH, fileName=obj_path,
                     meshScale=[1,1,1], flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
        hull_body_ids.append(p.createMultiBody(baseMass=0,
                     baseCollisionShapeIndex=col_id, basePosition=[0,0,0]))
    except Exception as e:
        print(f"  Failed: {obj_path}: {e}")
print(f"[INFO] {len(hull_body_ids)} obstacle meshes loaded")

# ══════════════════════════════════════════════════════════════════
# SDF
# ══════════════════════════════════════════════════════════════════

_sdf_cache = {}
_sdf_query_count = 0

def eval_sdf(q1, q2, q3):
    global _sdf_query_count
    key = (round(float(q1),4), round(float(q2),4), round(float(q3),4))
    if key in _sdf_cache: return _sdf_cache[key]
    p.resetJointState(arm3Id, 0, float(q1))
    p.resetJointState(arm3Id, 1, float(q2))
    p.resetJointState(arm3Id, 2, float(q3))
    p.stepSimulation()
    min_d = 10.0
    for hid in hull_body_ids:
        contacts = p.getClosestPoints(bodyA=arm3Id, bodyB=hid, distance=10.0)
        if contacts:
            min_d = min(min_d, min(c[8] for c in contacts))
    _sdf_cache[key] = min_d
    _sdf_query_count += 1
    return min_d

def eval_sdf_vec(cfg): return eval_sdf(cfg[0], cfg[1], cfg[2])

# ══════════════════════════════════════════════════════════════════
# START / GOAL
# ══════════════════════════════════════════════════════════════════

GOAL  = np.array([0.0, 0.0, 0.0])
START = np.array([0.0, math.pi / 2, 0.0])
Q_MIN, Q_MAX = -math.pi, math.pi

# ══════════════════════════════════════════════════════════════════
# FRONTIER GROWTH
# ══════════════════════════════════════════════════════════════════

LINE_SAMPLES  = 11
N_RAYS        = 8
RAY_LEN       = math.sqrt(2) * math.pi
N_RAY_SAMPLES = 12
t_vals_ray    = np.linspace(0.0, RAY_LEN, N_RAY_SAMPLES)
N_PTS_TARGET  = 20
N_BRANCHES    = 4

rng = np.random.default_rng(seed=42)

def random_unit_dirs(n):
    raw = rng.standard_normal((n, 3))
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)

def regula_falsi(a, b, fa, fb, tol=1e-4, max_iter=50):
    a, b, fa, fb = a.copy(), b.copy(), float(fa), float(fb)
    c = a.copy()
    for _ in range(max_iter):
        c  = a + fa * (a - b) / (fb - fa)
        fc = eval_sdf_vec(c)
        if abs(fc) < tol: return c
        if fa * fc < 0: b, fb = c, fc
        else:           a, fa = c, fc
    return c

def t_to_boundary(src, d):
    t_min, hit = RAY_LEN, False
    for dim in range(3):
        if   d[dim] > 0: t = (Q_MAX - src[dim]) / d[dim]
        elif d[dim] < 0: t = (Q_MIN - src[dim]) / d[dim]
        else: continue
        if 0 < t < t_min: t_min, hit = t, True
    return t_min if hit else None

def scatter_crossings(src):
    found = []
    for d in random_unit_dirs(N_RAYS):
        t_bnd    = t_to_boundary(src, d)
        prev_cfg = prev_sdf = None
        for t in t_vals_ray:
            if t_bnd is not None and t >= t_bnd:
                found.append((src + src + t_bnd * d) / 2.0)
                break
            cfg = src + t * d
            sdf = eval_sdf_vec(cfg)
            if prev_cfg is not None and prev_sdf * sdf < 0:
                found.append(regula_falsi(prev_cfg, cfg, prev_sdf, sdf))
                break
            prev_cfg, prev_sdf = cfg, sdf
    return found

def neg_sdf_midpoints(bpts):
    out = []
    for a, b in itertools.combinations(bpts, 2):
        m = (a + b) / 2.0
        s = eval_sdf_vec(m)
        if s < 0: out.append((m, s))
    return out

print(f"\n[INFO] Scanning START→GOAL … [{elapsed()}]")
line_cfgs = [START + t*(GOAL-START) for t in np.linspace(0,1,LINE_SAMPLES)]
line_sdfs = [eval_sdf_vec(c) for c in line_cfgs]
bpts = []
for i in range(LINE_SAMPLES-1):
    if line_sdfs[i]*line_sdfs[i+1] < 0:
        bpts.append(regula_falsi(line_cfgs[i], line_cfgs[i+1], line_sdfs[i], line_sdfs[i+1]))

root_cfg = None
for i in range(len(bpts)-1):
    m = (bpts[i]+bpts[i+1])/2.0
    if eval_sdf_vec(m) < 0: root_cfg = m
if root_cfg is None:
    raise RuntimeError("[FATAL] No negative-SDF midpoint on START→GOAL.")

r1_bpts  = scatter_crossings(root_cfg)
r1_cands = neg_sdf_midpoints(r1_bpts)
r1_cands.sort(key=lambda mc: -np.linalg.norm(mc[0]-root_cfg))
named_points = [root_cfg]
branch_tips  = [mc for mc,_ in r1_cands[:N_BRANCHES]]
named_points.extend(branch_tips)

round_num = 1
while len(named_points) < N_PTS_TARGET and branch_tips:
    round_num += 1
    next_tips = []
    for tip in branch_tips:
        if len(named_points) >= N_PTS_TARGET:
            next_tips.append(tip); continue
        cands = neg_sdf_midpoints(scatter_crossings(tip))
        if not cands:
            next_tips.append(tip); continue
        best, _ = max(cands, key=lambda mc: np.linalg.norm(mc[0]-tip))
        named_points.append(best)
        next_tips.append(best)
        print(f"  [round {round_num:03d}] {len(named_points)}/{N_PTS_TARGET}")
    branch_tips = next_tips

print(f"[INFO] Growth done — {len(named_points)} pts, {_sdf_query_count} SDF queries [{elapsed()}]")

# ══════════════════════════════════════════════════════════════════
# TETRAHEDRALISE C-SPACE
# ══════════════════════════════════════════════════════════════════

GRID_STEP = 0.1
q_vals = np.arange(-np.pi, np.pi + GRID_STEP * 0.5, GRID_STEP)
N = len(q_vals)
print(f"\n[INFO] Grid: {N}^3  [{elapsed()}]")

def vidx(i, j, k): return i*N*N + j*N + k

vertices = np.array([[q_vals[i], q_vals[j], q_vals[k]]
                     for i in range(N) for j in range(N) for k in range(N)])

def cube_tets(i, j, k):
    v = [vidx(i+di, j+dj, k+dk) for di,dj,dk in [
        (0,0,0),(1,0,0),(0,1,0),(1,1,0),(0,0,1),(1,0,1),(0,1,1),(1,1,1)]]
    if (i+j+k) % 2 == 0:
        return [[v[0],v[1],v[2],v[4]],[v[1],v[2],v[3],v[7]],
                [v[1],v[4],v[5],v[7]],[v[2],v[4],v[6],v[7]],[v[1],v[2],v[4],v[7]]]
    else:
        return [[v[0],v[1],v[3],v[5]],[v[0],v[2],v[3],v[6]],
                [v[0],v[4],v[5],v[6]],[v[3],v[5],v[6],v[7]],[v[0],v[3],v[5],v[6]]]

tetrahedra = np.array([tet for i in range(N-1) for j in range(N-1)
                       for k in range(N-1) for tet in cube_tets(i,j,k)], dtype=np.int32)
tet_centroids = vertices[tetrahedra].mean(axis=1)
n_tets = len(tetrahedra)
print(f"[INFO] {len(vertices):,} vertices, {n_tets:,} tets  [{elapsed()}]")

# ── Tet adjacency ──
print(f"[INFO] Building tet adjacency … [{elapsed()}]")
face_local_indices = np.array([[1,2,3],[0,2,3],[0,1,3],[0,1,2]], dtype=np.int32)
all_faces_sorted = np.sort(tetrahedra[:, face_local_indices].reshape(-1,3), axis=1)
SHIFT = int(np.ceil(np.log2(len(vertices)+1)))
v0,v1,v2 = (all_faces_sorted[:,c].astype(np.int64) for c in range(3))
face_keys = v0*(2**SHIFT)**2 + v1*(2**SHIFT) + v2
ti_arr = np.repeat(np.arange(n_tets, dtype=np.int64), 4)
order = np.argsort(face_keys, kind='stable')
sorted_keys = face_keys[order]; sorted_ti = ti_arr[order]
fi_arr = np.tile(np.arange(4, dtype=np.int64), n_tets)
sorted_fi = fi_arr[order]
match_idx = np.where(sorted_keys[:-1] == sorted_keys[1:])[0]
tet_nb = np.full((n_tets, 4), -1, dtype=np.int64)
tet_nb[sorted_ti[match_idx],   sorted_fi[match_idx]]   = sorted_ti[match_idx+1]
tet_nb[sorted_ti[match_idx+1], sorted_fi[match_idx+1]] = sorted_ti[match_idx]
print(f"[INFO] Adjacency done  [{elapsed()}]")

# ── Spatial lookup ──
def find_containing_tet(point):
    q = np.asarray(point, dtype=np.float64)
    result = []
    for dim in range(3):
        val = float(q[dim])
        if   val <= q_vals[0]:  result.append(0)
        elif val >= q_vals[-1]: result.append(N-2)
        else: result.append(min(int(np.searchsorted(q_vals, val, side='right')-1), N-2))
    i, j, k = result
    base = (i*(N-1)*(N-1) + j*(N-1) + k) * 5
    best_ti, best_min_b = base, -np.inf
    for ti in range(base, base+5):
        verts = vertices[tetrahedra[ti]]
        T = (verts[1:] - verts[0]).T
        try: lam = np.linalg.solve(T, q - verts[0])
        except: continue
        b = np.array([1-lam.sum(), lam[0], lam[1], lam[2]])
        mb = b.min()
        if mb >= -1e-9: return ti
        if mb > best_min_b: best_min_b, best_ti = mb, ti
    return best_ti

# ══════════════════════════════════════════════════════════════════
# CONVEX HULL
# ══════════════════════════════════════════════════════════════════

pts = np.array(named_points, dtype=np.float64)
print(f"\n[INFO] Computing convex hull over {len(pts)} points … [{elapsed()}]")
hull = ConvexHull(pts)
print(f"[INFO] Hull: {len(hull.vertices)} vertices, {len(hull.simplices)} faces")

hull_edges = set()
for simplex in hull.simplices:
    a, b, c = simplex
    for ea, eb in [(a,b),(b,c),(a,c)]:
        hull_edges.add((min(ea,eb), max(ea,eb)))
hull_edges = list(hull_edges)
print(f"[INFO] Hull edges: {len(hull_edges)}")

# ══════════════════════════════════════════════════════════════════
# SEGMENT TRAVERSAL
# Walk a→b through the tet mesh face by face using the adjacency
# table. At each tet, test all 4 faces for the exit: find the face
# whose plane the ray crosses next (smallest t > t_cur). Cross into
# the neighbour and repeat until t >= 1 (b reached) or grid boundary.
# ══════════════════════════════════════════════════════════════════

def tets_along_segment(a, b):
    ti      = find_containing_tet(a)
    end_ti  = find_containing_tet(b)
    d       = b - a
    visited = []
    seen    = set()
    t_cur   = 0.0
    prev_ti = -1

    for _ in range(50000):
        if ti < 0 or ti in seen: break
        seen.add(ti); visited.append(ti)
        if ti == end_ti: break

        verts     = vertices[tetrahedra[ti]]
        best_t    = np.inf
        best_face = -1

        for fi in range(4):
            if tet_nb[ti, fi] == prev_ti: continue
            fv = verts[[j for j in range(4) if j != fi]]
            v0, v1, v2 = fv
            normal = np.cross(v1-v0, v2-v0)
            denom  = normal @ d
            if abs(denom) < 1e-14: continue
            t = (normal @ (v0 - a)) / denom
            if t_cur + 1e-9 < t < best_t:
                best_t, best_face = t, fi

        if best_face < 0 or best_t > 1.0 + 1e-9: break
        nb = tet_nb[ti, best_face]
        if nb < 0: break
        prev_ti, t_cur, ti = ti, best_t, nb

    return visited

# ══════════════════════════════════════════════════════════════════
# TRIANGLE FLOOD FILL
#
# Given a hull triangle (V, A, B) in C-space, we want every tet
# whose centroid lies ON or INSIDE the triangle plane-slab.
#
# Step 1 — build the triangle's coordinate frame:
#   • normal n̂ = (AB × AV) / |…|   (perpendicular to the triangle)
#   • OOP_TOL = GRID_STEP * 0.6     (half-thickness of the slab;
#     just over half a cell so every tet touching the plane is caught)
#
# Step 2 — centroid_qualifies(ti):
#   For tet ti, project its centroid P onto the triangle plane:
#     d_oop = (P-A)·n̂              (out-of-plane distance)
#   Reject if |d_oop| > OOP_TOL.
#   Then compute barycentric coords inside the triangle:
#     Q = (P-A) - d_oop·n̂          (in-plane vector from A)
#     w_V, w_B solved via dot products with (AB×AV) system
#     w_A = 1 - w_V - w_B
#   Accept if w_V, w_B, w_A all ≥ -1e-9  (point inside triangle).
#
# Step 3 — BFS through tet adjacency:
#   Seed by sampling many points along all 3 edges + centroid of
#   the triangle, finding their containing tets, and keeping any
#   that qualify. Then BFS: for each accepted tet, check all 4
#   face-neighbours; if a neighbour qualifies, add it and enqueue.
#   This is efficient because the BFS only spreads within the slab —
#   tets outside the plane or outside the triangle boundary stop it.
# ══════════════════════════════════════════════════════════════════

OOP_TOL = GRID_STEP * 0.6   # slab half-thickness

def triangle_flood_fill(V, A, B):
    AB  = B - A
    AV  = V - A
    n   = np.cross(AB, AV)
    n_len = np.linalg.norm(n)
    if n_len < 1e-10: return set()
    n_hat   = n / n_len
    n_dot_n = np.dot(n, n)   # |n|^2, used for barycentric solve

    def qualifies(ti):
        AP   = tet_centroids[ti] - A
        d_oop = abs(np.dot(AP, n_hat))
        if d_oop > OOP_TOL: return False
        # in-plane barycentric coords
        Q    = AP - np.dot(AP, n_hat) * n_hat
        w_V  = np.dot(np.cross(AB, Q), n) / n_dot_n
        w_B  = np.dot(np.cross(Q, AV), n) / n_dot_n
        w_A  = 1.0 - w_V - w_B
        return w_V >= -1e-9 and w_B >= -1e-9 and w_A >= -1e-9

    # Seed: sample along all 3 edges + centroid
    visited, queue = set(), deque()
    N_SAMP = 12
    ts = np.linspace(0.0, 1.0, N_SAMP)
    seed_pts = (
        [V + t*(A-V) for t in ts] +
        [A + t*(B-A) for t in ts] +
        [V + t*(B-V) for t in ts] +
        [(V+A+B)/3.0]
    )
    for sp in seed_pts:
        ti = find_containing_tet(sp)
        if ti not in visited and qualifies(ti):
            visited.add(ti); queue.append(ti)

    if not queue: return set()

    # BFS through adjacency
    while queue:
        ti = queue.popleft()
        for fi in range(4):
            nb = tet_nb[ti, fi]
            if nb == -1 or nb in visited: continue
            if qualifies(nb):
                visited.add(nb); queue.append(nb)

    return visited

# ══════════════════════════════════════════════════════════════════
# SHADE TETS: EDGES + FACES
# ══════════════════════════════════════════════════════════════════

shaded_tets = set()

# ── Corner tets (vertex seeds) ──
for simplex in hull.simplices:
    for vi in simplex:
        shaded_tets.add(find_containing_tet(pts[vi]))

# ── Edge traversal ──
print(f"\n[INFO] Traversing {len(hull_edges)} hull edges … [{elapsed()}]")
for ia, ib in hull_edges:
    for ti in tets_along_segment(pts[ia], pts[ib]):
        shaded_tets.add(ti)
print(f"  After edges: {len(shaded_tets):,} tets  [{elapsed()}]")

# ── Face flood fill ──
print(f"[INFO] Flood-filling {len(hull.simplices)} hull faces … [{elapsed()}]")
for simplex in hull.simplices:
    ia, ib, ic = simplex
    for ti in triangle_flood_fill(pts[ia], pts[ib], pts[ic]):
        shaded_tets.add(ti)
print(f"  After faces: {len(shaded_tets):,} tets  [{elapsed()}]")

# ══════════════════════════════════════════════════════════════════
# CLASSIFY: centroid SDF
# ══════════════════════════════════════════════════════════════════

print(f"[INFO] Classifying {len(shaded_tets):,} tets by SDF … [{elapsed()}]")
red_tets, green_tets = [], []
for ti in shaded_tets:
    s = eval_sdf_vec(tet_centroids[ti])
    (red_tets if s < 0 else green_tets).append(ti)
print(f"  RED={len(red_tets):,}  GREEN={len(green_tets):,}  [{elapsed()}]")

p.disconnect()

# ══════════════════════════════════════════════════════════════════
# BUILD PYVISTA MESHES
# ══════════════════════════════════════════════════════════════════

def build_tet_mesh(index_list):
    if not index_list: return None
    idx  = np.array(index_list, dtype=np.int64)
    tets = tetrahedra[idx]; M = len(idx)
    unique_vids, local_tets = np.unique(tets, return_inverse=True)
    local_tets  = local_tets.reshape(M, 4)
    local_verts = np.degrees(vertices[unique_vids])
    cells  = np.hstack([np.full((M,1),4,dtype=np.int64), local_tets]).ravel()
    ctypes = np.full(M, 10, dtype=np.uint8)
    return pv.UnstructuredGrid(cells, ctypes, local_verts)

red_mesh   = build_tet_mesh(red_tets)
green_mesh = build_tet_mesh(green_tets)

pts_deg  = np.degrees(pts)
edge_pts = []
for ia, ib in hull_edges:
    edge_pts += [pts_deg[ia], pts_deg[ib]]
edge_pts = np.array(edge_pts); n_e = len(edge_pts)//2
hull_wire = pv.PolyData(edge_pts)
hull_wire.lines = np.hstack([np.full((n_e,1),2),
                              np.arange(n_e*2).reshape(n_e,2)]).ravel()

sg_line = pv.Spline(np.degrees(np.array(
    [START+t*(GOAL-START) for t in np.linspace(0,1,200)])), 200)

# ══════════════════════════════════════════════════════════════════
# VISUALISE
# ══════════════════════════════════════════════════════════════════

pl = pv.Plotter(window_size=[1400, 900])
pl.set_background("#1a1a2e")

pl.add_mesh(sg_line, color="dodgerblue", line_width=2, opacity=0.5)
pl.add_points(np.degrees(START).reshape(1,3),    color="lime",    point_size=20, render_points_as_spheres=True)
pl.add_points(np.degrees(GOAL).reshape(1,3),     color="orange",  point_size=20, render_points_as_spheres=True)
pl.add_points(np.degrees(root_cfg).reshape(1,3), color="magenta", point_size=20, render_points_as_spheres=True)
pl.add_points(pts_deg, color="#9B59B6", point_size=10, render_points_as_spheres=True)

actors = {}
vis = {'red': True, 'green': True, 'hull': True, 'hverts': True}

if red_mesh:
    actors['red']   = pl.add_mesh(red_mesh,   color="#E74C3C", opacity=0.7,
                                   show_edges=True, edge_color="#C0392B", line_width=0.5)
if green_mesh:
    actors['green'] = pl.add_mesh(green_mesh, color="#2ECC71", opacity=0.7,
                                   show_edges=True, edge_color="#27AE60", line_width=0.5)
actors['hull']   = pl.add_mesh(hull_wire, color="#F39C12", line_width=2, opacity=0.9)
actors['hverts'] = pl.add_points(pts_deg[hull.vertices], color="white",
                                  point_size=14, render_points_as_spheres=True)

def toggle(key):
    vis[key] = not vis[key]
    if key in actors: actors[key].SetVisibility(vis[key])
    pl.update()

pl.add_key_event('r', lambda: toggle('red'))
pl.add_key_event('g', lambda: toggle('green'))
pl.add_key_event('h', lambda: toggle('hull'))
pl.add_key_event('v', lambda: toggle('hverts'))

pl.add_axes(xlabel="q1 (deg)", ylabel="q2 (deg)", zlabel="q3 (deg)")
pl.add_title(
    f"Edges+Faces shaded  |  {len(pts)} pts  |  {len(hull.simplices)} faces  "
    f"|  {len(shaded_tets):,} tets  |  RED={len(red_tets):,}  GREEN={len(green_tets):,}\n"
    f"Keys: [r] red  [g] green  [h] hull wire  [v] hull vertices",
    font_size=10, color="white")

print(f"\n[INFO] Total pre-viz time: {elapsed()}")
print("  [r] red  [g] green  [h] hull  [v] verts")
pl.show(interactive=True, auto_close=False)
print("[INFO] Done.")