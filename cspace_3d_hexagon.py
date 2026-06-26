import pybullet_data
import numpy as np
import pybullet as p
import time, os, glob, math
from collections import defaultdict, deque
import pyvista as pv

_SCRIPT_START    = time.perf_counter()  # ← add
_sdf_query_count = 0                    # ← add

# ══════════════════════════════════════════════════════════════════
# PYBULLET SETUP
# ══════════════════════════════════════════════════════════════════

physicsClient = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
planeId = p.loadURDF("plane.urdf")

arm3Id = p.loadURDF(
    "arm_3.urdf", basePosition=[0, 0, 0],
    baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
    useFixedBase=True,
    flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION)
NUM_JOINTS = p.getNumJoints(arm3Id)
print(f"[INFO] 3-DOF arm loaded — {NUM_JOINTS} joints")

hull_body_ids = []
for obj_path in sorted(glob.glob(os.path.join("objs", "*.obj"))):
    try:
        col_id = p.createCollisionShape(
            p.GEOM_MESH, fileName=obj_path,
            meshScale=[1, 1, 1],
            flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
        body_id = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=col_id,
            basePosition=[0, 0, 0])
        hull_body_ids.append(body_id)
    except Exception as e:
        print(f"Failed to load {obj_path}: {e}")

print(f"[INFO] {len(hull_body_ids)} mesh bodies ready")

# ══════════════════════════════════════════════════════════════════
# SDF
# ══════════════════════════════════════════════════════════════════

_sdf_cache = {}

def sdf_scene(arm_id, threshold=10.0):
    min_d = threshold
    for hull_id in hull_body_ids:
        contacts = p.getClosestPoints(bodyA=arm_id, bodyB=hull_id, distance=threshold)
        if contacts:
            d = min(c[8] for c in contacts)
            if d < min_d:
                min_d = d
    return min_d

def set_config(q1, q2, q3):
    p.resetJointState(arm3Id, 0, float(q1))
    p.resetJointState(arm3Id, 1, float(q2))
    p.resetJointState(arm3Id, 2, float(q3))
    p.stepSimulation()

def eval_sdf(q1, q2, q3):
    global _sdf_query_count
    key = (round(float(q1), 4), round(float(q2), 4), round(float(q3), 4))
    if key in _sdf_cache:
        return _sdf_cache[key]
    set_config(q1, q2, q3)
    d = sdf_scene(arm3Id)
    _sdf_cache[key] = d
    _sdf_query_count += 1  # ← only counts cache misses (real PyBullet calls)
    return d

def eval_sdf_vec(cfg):
    return eval_sdf(cfg[0], cfg[1], cfg[2])

# ══════════════════════════════════════════════════════════════════
# START / GOAL
# ══════════════════════════════════════════════════════════════════

GOAL  = np.array([0.0, 0.0, 0.0])
START = np.array([0.0, math.pi / 2, 0.0])

# ══════════════════════════════════════════════════════════════════
# TETRAHEDRALISE C-SPACE
# ══════════════════════════════════════════════════════════════════

GRID_STEP = 0.1
q_vals = np.arange(-np.pi, np.pi + GRID_STEP * 0.5, GRID_STEP)
N = len(q_vals)
print(f"[INFO] Grid: {N}x{N}x{N} nodes (step={GRID_STEP:.2f} rad)")

def vidx(i, j, k):
    return i * N * N + j * N + k

vertices = np.array([[q_vals[i], q_vals[j], q_vals[k]]
                     for i in range(N) for j in range(N) for k in range(N)])

def cube_tets(i, j, k):
    v = [vidx(i+di, j+dj, k+dk) for di, dj, dk in [
        (0,0,0),(1,0,0),(0,1,0),(1,1,0),
        (0,0,1),(1,0,1),(0,1,1),(1,1,1)
    ]]
    if (i + j + k) % 2 == 0:
        return [
            [v[0],v[1],v[2],v[4]],
            [v[1],v[2],v[3],v[7]],
            [v[1],v[4],v[5],v[7]],
            [v[2],v[4],v[6],v[7]],
            [v[1],v[2],v[4],v[7]],
        ]
    else:
        return [
            [v[0],v[1],v[3],v[5]],
            [v[0],v[2],v[3],v[6]],
            [v[0],v[4],v[5],v[6]],
            [v[3],v[5],v[6],v[7]],
            [v[0],v[3],v[5],v[6]],
        ]

tetrahedra = np.array([
    tet
    for i in range(N-1) for j in range(N-1) for k in range(N-1)
    for tet in cube_tets(i, j, k)
], dtype=np.int32)

tet_centroids = vertices[tetrahedra].mean(axis=1)
print(f"[INFO] {len(vertices):,} vertices, {len(tetrahedra):,} tetrahedra")

# ══════════════════════════════════════════════════════════════════
# PRECOMPUTE TET ADJACENCY
# ══════════════════════════════════════════════════════════════════

print("[INFO] Building tet adjacency …")
t0 = time.perf_counter()

LOCAL_FACES = [(1,2,3), (0,2,3), (0,1,3), (0,1,2)]
n_tets = len(tetrahedra)

# Extract all 4 faces per tet as sorted vertex triples — shape (n_tets*4, 3)
face_local_indices = np.array([[1,2,3],[0,2,3],[0,1,3],[0,1,2]], dtype=np.int32)
all_faces        = tetrahedra[:, face_local_indices].reshape(-1, 3)  # (n_tets*4, 3)
all_faces_sorted = np.sort(all_faces, axis=1)                        # canonical order

# Encode each face as a single int64 for fast sorting/matching
SHIFT    = int(np.ceil(np.log2(len(vertices) + 1)))
v0, v1, v2 = (all_faces_sorted[:, c].astype(np.int64) for c in range(3))
face_keys   = v0 * (2**SHIFT)**2 + v1 * (2**SHIFT) + v2

ti_arr = np.repeat(np.arange(n_tets, dtype=np.int64), 4)      # tet index per row
fi_arr = np.tile  (np.arange(4,      dtype=np.int64), n_tets)  # face index per row

# Sort by face key — shared faces become adjacent pairs
order        = np.argsort(face_keys, kind='stable')
sorted_keys  = face_keys[order]
sorted_ti    = ti_arr[order]
sorted_fi    = fi_arr[order]

# Consecutive equal keys → shared face between two tets
match_idx = np.where(sorted_keys[:-1] == sorted_keys[1:])[0]

tet_neighbours = np.full((n_tets, 4), -1, dtype=np.int64)
ti0 = sorted_ti[match_idx];     fi0 = sorted_fi[match_idx]
ti1 = sorted_ti[match_idx + 1]; fi1 = sorted_fi[match_idx + 1]
tet_neighbours[ti0, fi0] = ti1
tet_neighbours[ti1, fi1] = ti0

print(f"[INFO] Adjacency done in {time.perf_counter()-t0:.2f}s")

# ══════════════════════════════════════════════════════════════════
# SPATIAL LOOKUP: point → tet
# ══════════════════════════════════════════════════════════════════

def find_tet_containing(point, tol=1e-9):
    q = np.asarray(point, dtype=np.float64)
    q_clamped = np.clip(q, q_vals[0], q_vals[-2])
    i = int(np.searchsorted(q_vals, q_clamped[0], side='right') - 1)
    j = int(np.searchsorted(q_vals, q_clamped[1], side='right') - 1)
    k = int(np.searchsorted(q_vals, q_clamped[2], side='right') - 1)
    i = min(i, N-2); j = min(j, N-2); k = min(k, N-2)
    cube_flat = (i*(N-1)*(N-1) + j*(N-1) + k) * 5
    for ti in range(cube_flat, cube_flat + 5):
        verts = vertices[tetrahedra[ti]]
        T = (verts[1:] - verts[0]).T
        try:
            lam = np.linalg.solve(T, q - verts[0])
        except np.linalg.LinAlgError:
            continue
        lam0 = 1.0 - lam.sum()
        bary = np.array([lam0, lam[0], lam[1], lam[2]])
        if np.all(bary >= -tol):
            return ti
    return cube_flat

# ══════════════════════════════════════════════════════════════════
# SEGMENT TRAVERSAL
# ══════════════════════════════════════════════════════════════════

def segment_exit_face(tet_idx, origin, direction, t_min, t_max):
    tet   = tetrahedra[tet_idx]
    verts = vertices[tet]
    best_t    = t_max
    best_face = None
    for fi, (a, b, c) in enumerate(LOCAL_FACES):
        opp_local = list({0,1,2,3} - {a,b,c})[0]
        Va = verts[a]; Vb = verts[b]; Vc = verts[c]
        Vopp = verts[opp_local]
        normal = np.cross(Vb - Va, Vc - Va)
        if np.dot(normal, Vopp - Va) < 0:
            normal = -normal
        n_out = -normal
        denom = np.dot(n_out, direction)
        if denom <= 1e-12:
            continue
        t_hit = np.dot(n_out, Va - origin) / denom
        if t_min + 1e-10 < t_hit < best_t + 1e-10:
            best_t    = t_hit
            best_face = fi
    return best_face, best_t

def traverse_segment(A, B, max_steps=50000):
    direction = B - A
    seg_len   = np.linalg.norm(direction)
    if seg_len < 1e-12:
        return set()
    direction = direction / seg_len
    t_end     = seg_len
    visited   = set()
    ti        = find_tet_containing(A)
    t_cur     = 0.0
    for _ in range(max_steps):
        visited.add(ti)
        if t_cur >= t_end - 1e-10:
            break
        exit_fi, t_exit = segment_exit_face(ti, A, direction, t_cur, t_end)
        if exit_fi is None:
            break
        nb = tet_neighbours[ti][exit_fi]
        if nb == -1:
            break
        t_cur = t_exit
        ti    = nb
    return visited

# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

def regula_falsi_3d(cfg_a, cfg_b, sdf_a, sdf_b, tol=1e-4, max_iter=50):
    a, b   = cfg_a.copy(), cfg_b.copy()
    fa, fb = float(sdf_a), float(sdf_b)
    for _ in range(max_iter):
        cfg_c = a + fa * (a - b) / (fb - fa)
        fc    = eval_sdf_vec(cfg_c)
        if abs(fc) < tol:
            return cfg_c, fc
        if fa * fc < 0:
            b, fb = cfg_c, fc
        else:
            a, fa = cfg_c, fc
    return cfg_c, fc

C_SPACE_LIMIT = np.pi

def _hits_extremity(cfg):
    return bool(np.any(np.abs(cfg) >= C_SPACE_LIMIT))

def ray_midpoint(origin, direction, ray_len, n_samples=11):
    t_vals = np.linspace(0.0, ray_len, n_samples)
    cfgs   = [origin + t * direction for t in t_vals]
    sdfs   = [eval_sdf_vec(c) for c in cfgs]
    crossings = []
    for i in range(n_samples - 1):
        if len(crossings) == 2:
            break
        if sdfs[i] * sdfs[i+1] < 0:
            # Linear interpolation — no extra SDF calls, accurate enough
            # for node placement (we just need a point near the boundary)
            t_cross = sdfs[i] / (sdfs[i] - sdfs[i+1])
            cfg_z = cfgs[i] + t_cross * (cfgs[i+1] - cfgs[i])
            crossings.append(cfg_z)
            continue
        if not _hits_extremity(cfgs[i]) and _hits_extremity(cfgs[i+1]):
            ta, tb = t_vals[i], t_vals[i+1]
            for _ in range(30):
                tm = (ta + tb) / 2.0
                if _hits_extremity(origin + tm * direction):
                    tb = tm
                else:
                    ta = tm
            crossings.append(origin + ta * direction)
    if len(crossings) < 2:
        return None, None, crossings
    mid_cfg = (crossings[0] + crossings[1]) / 2.0
    mid_sdf = eval_sdf_vec(mid_cfg)
    return mid_cfg, mid_sdf, crossings

# ══════════════════════════════════════════════════════════════════
# GRAPH DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════

HEX_SIDE_LEN  = 1.0
RAY_LEN       = math.sqrt(2) * math.pi
N_RAY_SAMPLES = 11
MERGE_THRESH  = 0.2
N_LAYERS      = 4

graph_nodes   = {}
graph_edges   = set()
layer_map     = {}
recorded_rays = []

def record_ray(origin, direction, ray_len, crossings, midpoint):
    recorded_rays.append({
        'origin':    np.array(origin),
        'end':       np.array(origin + ray_len * direction),
        'crossings': [np.array(c) for c in crossings],
        'midpoint':  np.array(midpoint) if midpoint is not None else None,
    })

def add_node(label, cfg, sdf, layer):
    graph_nodes[label] = {
        'cfg':          np.array(cfg),
        'sdf':          float(sdf),
        'in_collision': sdf < 0,
        'label':        label,
        'neighbours':   set(),
    }
    layer_map[label] = layer

def add_edge(la, lb):
    if la == lb:
        return
    graph_edges.add(frozenset({la, lb}))
    graph_nodes[la]['neighbours'].add(lb)
    graph_nodes[lb]['neighbours'].add(la)

def merge_check_round(new_configs):
    kept = []
    for item in new_configs:
        label, cfg, sdf = item
        too_close = False
        for _, k_cfg, _ in kept:
            if np.linalg.norm(cfg - k_cfg) < MERGE_THRESH:
                too_close = True
                break
        if not too_close:
            kept.append(item)
    return kept

_label_counter = [0]

def _next_label(prefix):
    _label_counter[0] += 1
    return f"{prefix}_{_label_counter[0]}"

def expand_node(Pp, Pc_label, child_prefix, layer,
                ray_len=RAY_LEN, n_samples=N_RAY_SAMPLES):
    Pc   = graph_nodes[Pc_label]['cfg']
    axis = Pc - Pp
    dist = np.linalg.norm(axis)
    if dist < 1e-8:
        return []
    axis_n = axis / dist
    if abs(axis_n[0]) < 0.9:
        tmp = np.array([1.0, 0.0, 0.0])
    else:
        tmp = np.array([0.0, 1.0, 0.0])
    u = np.cross(axis_n, tmp);  u /= np.linalg.norm(u)
    v = np.cross(axis_n, u);    v /= np.linalg.norm(v)

    all_angles = [2 * math.pi * k / 6 for k in range(6)]
    existing_neighbours = list(graph_nodes[Pc_label]['neighbours'])
    hex_neighbours = []
    for nb_label in existing_neighbours:
        nb_cfg = graph_nodes[nb_label]['cfg']
        delta  = nb_cfg - Pc
        proj_u = np.dot(delta, u)
        proj_v = np.dot(delta, v)
        if abs(proj_u) < 1e-6 and abs(proj_v) < 1e-6:
            continue
        angle = math.atan2(proj_v, proj_u) % (2 * math.pi)
        hex_neighbours.append((angle, nb_label))

    ANGLE_TOL = math.pi / 6
    covered_indices = set()
    for (nb_angle, _) in hex_neighbours:
        diffs = [min(abs(nb_angle - a) % (2*math.pi),
                     (2*math.pi) - abs(nb_angle - a) % (2*math.pi))
                 for a in all_angles]
        best = int(np.argmin(diffs))
        if diffs[best] < ANGLE_TOL:
            covered_indices.add(best)

    missing_indices = [i for i in range(6) if i not in covered_indices]
    if not missing_indices:
        return []

    new_raw = []
    for idx in missing_indices:
        angle   = all_angles[idx]
        rim     = math.cos(angle) * u + math.sin(angle) * v
        ray_dir = axis_n + (HEX_SIDE_LEN / dist) * rim
        ray_dir /= np.linalg.norm(ray_dir)
        cfg, sdf, crossings = ray_midpoint(Pp, ray_dir, ray_len, n_samples)
        record_ray(Pp, ray_dir, ray_len, crossings, cfg)
        if cfg is None:
            continue
        cand_label = _next_label(child_prefix)
        new_raw.append((cand_label, cfg, sdf))

    new_raw = merge_check_round(new_raw)
    new_labels = []
    for (label, cfg, sdf) in new_raw:
        add_node(label, cfg, sdf, layer)
        new_labels.append(label)

    for label in new_labels:
        add_edge(Pc_label, label)

    def node_angle(lbl):
        delta = graph_nodes[lbl]['cfg'] - Pc
        return math.atan2(np.dot(delta, v), np.dot(delta, u)) % (2 * math.pi)

    all_ring_labels = [nb for (_, nb) in hex_neighbours] + new_labels
    all_ring_labels.sort(key=node_angle)
    n_ring = len(all_ring_labels)
    for i in range(n_ring):
        add_edge(all_ring_labels[i], all_ring_labels[(i + 1) % n_ring])

    return new_labels

# ══════════════════════════════════════════════════════════════════
# INITIAL RAY → P00
# ══════════════════════════════════════════════════════════════════

sg_vec      = GOAL - START
sg_dist     = np.linalg.norm(sg_vec)
sg_dir      = sg_vec / sg_dist
t_vals_init = np.linspace(0.0, sg_dist, N_RAY_SAMPLES)

print(f"\n[INFO] Shooting P00 ray along START→GOAL …")
t0 = time.perf_counter()

cfgs_init = [START + t * sg_dir for t in t_vals_init]
sdfs_init = [eval_sdf_vec(c) for c in cfgs_init]

crossings_init = []
for i in range(N_RAY_SAMPLES - 1):
    if len(crossings_init) == 2:
        break
    if sdfs_init[i] * sdfs_init[i+1] < 0:
        t_cross = sdfs_init[i] / (sdfs_init[i] - sdfs_init[i+1])
        cfg_z   = cfgs_init[i] + t_cross * (cfgs_init[i+1] - cfgs_init[i])
        crossings_init.append(cfg_z)

P00_cfg = None
if len(crossings_init) >= 2:
    mid_cfg = (crossings_init[0] + crossings_init[1]) / 2.0
    mid_sdf = eval_sdf_vec(mid_cfg)
    if mid_sdf < 0:
        P00_cfg = mid_cfg

if P00_cfg is None:
    print("[ERROR] No collision found on START→GOAL line.")
    p.disconnect()
    raise SystemExit

add_node("P00", P00_cfg, eval_sdf_vec(P00_cfg), layer=0)
print(f"[INFO] P00 ray done in {time.perf_counter()-t0:.2f}s")

# ══════════════════════════════════════════════════════════════════
# PRUNING
# ══════════════════════════════════════════════════════════════════

def _remove_node(lbl):
    for nb in list(graph_nodes[lbl]['neighbours']):
        graph_nodes[nb]['neighbours'].discard(lbl)
        graph_edges.discard(frozenset({lbl, nb}))
    del graph_nodes[lbl]
    del layer_map[lbl]

def _same_layer_degree(lbl, layer_set):
    return len([nb for nb in graph_nodes[lbl]['neighbours'] if nb in layer_set])

def _nearest_unconnected(lbl, exclude_lbl, layer_set):
    cfg = graph_nodes[lbl]['cfg']
    best_dist, best_nb = float('inf'), None
    for cand in layer_set:
        if cand == lbl or cand == exclude_lbl:
            continue
        if cand in graph_nodes[lbl]['neighbours']:
            continue
        d = np.linalg.norm(graph_nodes[cand]['cfg'] - cfg)
        if d < best_dist:
            best_dist, best_nb = d, cand
    return best_nb

def prune_layer(layer_set):
    changed = False
    zero_deg, one_deg = [], []
    for lbl in list(layer_set):
        d = _same_layer_degree(lbl, layer_set)
        if d == 0:
            zero_deg.append(lbl)
        elif d == 1:
            one_deg.append(lbl)
    for lbl in zero_deg:
        if lbl not in layer_set:
            continue
        layer_set.discard(lbl)
        _remove_node(lbl)
        changed = True
    for lbl in one_deg:
        if lbl not in layer_set:
            continue
        if _same_layer_degree(lbl, layer_set) != 1:
            continue
        nb = next(n for n in graph_nodes[lbl]['neighbours'] if n in layer_set)
        nb_deg = _same_layer_degree(nb, layer_set)
        if nb_deg >= 3:
            layer_set.discard(lbl)
            _remove_node(lbl)
            changed = True
        elif nb_deg == 2:
            layer_set.discard(lbl)
            _remove_node(lbl)
            changed = True
            if nb in layer_set:
                nearest = _nearest_unconnected(nb, lbl, layer_set)
                if nearest is not None:
                    add_edge(nb, nearest)
    return changed

# ══════════════════════════════════════════════════════════════════
# LAYER EXPANSION LOOP
# ══════════════════════════════════════════════════════════════════

current_layer_labels = ["P00"]
for layer_idx in range(1, N_LAYERS + 1):
    child_prefix = f"P{layer_idx}"
    print(f"\n[INFO] ── Layer {layer_idx} ── expanding {len(current_layer_labels)} nodes …")
    next_layer_labels = []
    for lbl in current_layer_labels:
        new = expand_node(Pp=START, Pc_label=lbl,
                          child_prefix=child_prefix, layer=layer_idx)
        next_layer_labels.extend(new)
    next_layer_labels = list(dict.fromkeys(next_layer_labels))
    layer_set = set(next_layer_labels)
    pass_num  = 0
    while True:
        pass_num += 1
        if not prune_layer(layer_set):
            break
    next_layer_labels = list(layer_set)
    current_layer_labels = next_layer_labels
    if not current_layer_labels:
        break

print(f"\n[INFO] Graph: {len(graph_nodes)} nodes, {len(graph_edges)} edges")
# NOTE: p.disconnect() moved to after centroid SDF evaluation below

# ══════════════════════════════════════════════════════════════════
# SHADE TETS VIA EDGE TRAVERSAL + TRIANGLE WEDGE FILL
# ══════════════════════════════════════════════════════════════════
#
# Strategy:
#   Step 1 — traverse each graph edge A→B, shade all pierced tets.
#
#   Step 2 — for each graph vertex V, sort its neighbours angularly
#             and process each consecutive pair (A, B) as a triangle
#             V-A-B:
#               • BFS-flood the tets inside that triangle's prism
#                 (bounded by the triangle itself, not an infinite wedge)
#               • add edge A-B to the graph if missing
#               • mark the triangle {V,A,B} as done so that when we
#                 later visit vertex A or B we skip the same triangle.
#
# The "inside triangle" test for a centroid P:
#   Project P onto the plane of V,A,B.  Express the projection as
#   barycentric coords (α,β,γ) w.r.t. the triangle.  Accept iff
#   α,β,γ ≥ 0  (i.e. inside or on the triangle).
#   Also cap the out-of-plane distance to a small fraction of the
#   triangle size so we don't pull in distant tets.
# ══════════════════════════════════════════════════════════════════

print("\n[INFO] Traversing graph edges through tet mesh …")
t0 = time.perf_counter()

shaded_tets = set()

# ── Step 1: shade tets pierced by each graph edge ─────────────────
for edge in graph_edges:
    la, lb = tuple(edge)
    A = graph_nodes[la]['cfg']
    B = graph_nodes[lb]['cfg']
    shaded_tets |= traverse_segment(A, B)

print(f"  After edge traversal: {len(shaded_tets):,} red tets  "
      f"({time.perf_counter()-t0:.1f}s)")

# ── Step 2: triangle wedge fill ────────────────────────────────────

def triangle_flood_fill(V, A, B):
    """
    Flood-fill tets whose centroid lies near the plane of triangle V-A-B,
    bounded within the triangle's barycentric extent.

    Algorithm:
      1. Find the tet containing the triangle centroid → seed.
      2. BFS: expand to face-adjacent neighbours whose centroid distance
         to the triangle plane is <= OOP_TOL  AND  whose centroid projects
         inside (or on) the triangle (barycentric check).
    """
    AB = B - A
    AV = V - A
    n  = np.cross(AB, AV)
    n_len = np.linalg.norm(n)
    if n_len < 1e-10:
        return set()
    n_hat   = n / n_len
    n_dot_n = np.dot(n, n)

    # Distance from triangle plane a tet centroid must be within
    OOP_TOL = GRID_STEP * 0.75   # slightly under one cell to avoid bleed

    def centroid_qualifies(ti):
        P  = tet_centroids[ti]
        AP = P - A
        # Out-of-plane distance
        if abs(np.dot(AP, n_hat)) > OOP_TOL:
            return False
        # Barycentric coords of in-plane projection
        Q   = AP - np.dot(AP, n_hat) * n_hat
        w_V = np.dot(np.cross(AB, Q), n) / n_dot_n
        w_B = np.dot(np.cross(Q, AV), n) / n_dot_n
        w_A = 1.0 - w_V - w_B
        return w_V >= -1e-9 and w_B >= -1e-9 and w_A >= -1e-9

    # Seed: tet containing the triangle centroid
    tri_centroid = (V + A + B) / 3.0
    seed = find_tet_containing(tri_centroid)

    visited = set()
    queue   = deque()

    if centroid_qualifies(seed):
        visited.add(seed)
        queue.append(seed)
    else:
        # Centroid tet didn't qualify (e.g. degenerate triangle) — skip
        return set()

    while queue:
        ti = queue.popleft()
        for fi in range(4):
            nb = tet_neighbours[ti][fi]
            if nb == -1 or nb in visited:
                continue
            if centroid_qualifies(nb):
                visited.add(nb)
                queue.append(nb)

    return visited


# Cache edge traversals so each edge is walked at most once
_edge_tet_cache = {}

def get_edge_tets(la, lb):
    key = frozenset({la, lb})
    if key not in _edge_tet_cache:
        A = graph_nodes[la]['cfg']
        B = graph_nodes[lb]['cfg']
        _edge_tet_cache[key] = traverse_segment(A, B)
    return _edge_tet_cache[key]

# Pre-populate cache from Step 1 traversals
for edge in graph_edges:
    la, lb = tuple(edge)
    key = frozenset({la, lb})
    # Already shaded — rebuild per-edge sets by re-traversing
    # (cheap since traverse_segment is fast and results are cached)
    _edge_tet_cache[key] = traverse_segment(
        graph_nodes[la]['cfg'], graph_nodes[lb]['cfg'])

# Visited triangles: frozenset of 3 node labels
done_triangles = set()
wedge_total    = 0

for v_label, node in list(graph_nodes.items()):
    neighbours = list(node['neighbours'])
    if len(neighbours) < 2:
        continue

    Vc = node['cfg']

    # Sort neighbours angularly in a stable 2-D frame
    dirs = []
    for nb in neighbours:
        d = graph_nodes[nb]['cfg'] - Vc
        dn = np.linalg.norm(d)
        if dn > 1e-8:
            dirs.append(d / dn)

    if len(dirs) < 2:
        continue

    ref_cross = np.cross(dirs[0], dirs[1])
    ref_len   = np.linalg.norm(ref_cross)
    plane_normal = ref_cross / ref_len if ref_len > 1e-8 else np.array([0.,0.,1.])

    if abs(plane_normal[0]) < 0.9:
        tmp = np.array([1.0, 0.0, 0.0])
    else:
        tmp = np.array([0.0, 1.0, 0.0])
    u_ax = np.cross(plane_normal, tmp);  u_ax /= np.linalg.norm(u_ax)
    v_ax = np.cross(plane_normal, u_ax); v_ax /= np.linalg.norm(v_ax)

    def angle_2d(nb_label):
        d = graph_nodes[nb_label]['cfg'] - Vc
        return math.atan2(float(np.dot(d, v_ax)), float(np.dot(d, u_ax)))

    neighbours_sorted = sorted(neighbours, key=angle_2d)
    n = len(neighbours_sorted)

    for i in range(n):
        nb_A = neighbours_sorted[i]
        nb_B = neighbours_sorted[(i + 1) % n]

        tri_key = frozenset({v_label, nb_A, nb_B})
        if tri_key in done_triangles:
            continue

        # Ensure edge A-B exists in the graph
        if nb_B not in graph_nodes[nb_A]['neighbours']:
            add_edge(nb_A, nb_B)

        # Get (cached) tet sets for all three edges of this triangle
        tets_VA = get_edge_tets(v_label, nb_A)
        tets_VB = get_edge_tets(v_label, nb_B)
        tets_AB = get_edge_tets(nb_A,    nb_B)

        V = graph_nodes[v_label]['cfg']
        A = graph_nodes[nb_A]['cfg']
        B = graph_nodes[nb_B]['cfg']

        new_tets = triangle_flood_fill(V, A, B)  # no edge tet args needed
        before = len(shaded_tets)
        shaded_tets |= new_tets
        wedge_total += len(shaded_tets) - before

        # Mark all three vertex-pair orderings as done
        done_triangles.add(tri_key)

print(f"  After triangle fill:  {len(shaded_tets):,} red tets  "
      f"(+{wedge_total:,} from triangles, {time.perf_counter()-t0:.1f}s)")

# ══════════════════════════════════════════════════════════════════
# BUILD PYVISTA MESHES
# ══════════════════════════════════════════════════════════════════

LAYER_COLOURS = ["#E74C3C","#E67E22","#F1C40F","#2ECC71","#3498DB","#9B59B6"]

print("[INFO] Building red tet mesh …")
t0 = time.perf_counter()

if shaded_tets:
    shaded_list = sorted(shaded_tets)
    print(f"  Classifying {len(shaded_list):,} shaded tets via cached vertex SDFs …")

    # FIX 1: Re-use SDF values already cached at grid vertices — no new PyBullet calls.
    # A tet is "in collision" if ANY of its 4 corner vertices has SDF < 0
    # (same sign convention as boundary BFS code).
    _vertex_sdf_hex = {}

    def _get_vsdf(vi):
        if vi not in _vertex_sdf_hex:
            _vertex_sdf_hex[vi] = eval_sdf_vec(vertices[vi])
        return _vertex_sdf_hex[vi]

    collision_list = []  # any vertex sdf < 0  → red (in/touching obstacle)
    free_list      = []  # all vertices sdf >= 0 → green (free space)

    for ti in shaded_list:
        tet_verts = tetrahedra[ti]
        sdfs = [_get_vsdf(vi) for vi in tet_verts]
        if any(s < 0 for s in sdfs):
            collision_list.append(ti)
        else:
            free_list.append(ti)

    print(f"  Collision (red): {len(collision_list):,}  |  "
          f"Free (green): {len(free_list):,}  "
          f"({time.perf_counter()-t0:.1f}s)")

    p.disconnect()

    def _build_tet_mesh(index_list):
        arr      = tetrahedra[index_list].astype(np.int64)
        n        = len(index_list)
        cells    = np.hstack([np.full((n, 1), 4, dtype=np.int64), arr]).ravel()
        ctypes   = np.full(n, 10, dtype=np.uint8)
        return pv.UnstructuredGrid(cells, ctypes, np.degrees(vertices))

    red_tet_mesh   = _build_tet_mesh(collision_list) if collision_list else None
    green_tet_mesh = _build_tet_mesh(free_list)      if free_list      else None
else:
    red_tet_mesh   = None
    green_tet_mesh = None
    print("  No shaded tets.")

# ── Edge mesh ──
edge_mesh = None
if graph_edges:
    edge_pts = []
    for edge in graph_edges:
        la, lb = tuple(edge)
        edge_pts.append(np.degrees(graph_nodes[la]['cfg']))
        edge_pts.append(np.degrees(graph_nodes[lb]['cfg']))
    edge_pts = np.array(edge_pts, dtype=np.float64)
    n_e      = len(edge_pts) // 2
    cells_e  = np.hstack([np.full((n_e,1), 2, dtype=np.int64),
                           np.arange(n_e*2, dtype=np.int64).reshape(n_e,2)]).ravel()
    edge_mesh = pv.PolyData(edge_pts)
    edge_mesh.lines = cells_e

# ── Node meshes per layer ──
node_meshes = {}
for layer_idx in range(N_LAYERS + 1):
    labels_in_layer = [lbl for lbl, l in layer_map.items() if l == layer_idx]
    col_pts, free_pts = [], []
    for lbl in labels_in_layer:
        nd  = graph_nodes[lbl]
        deg = np.degrees(nd['cfg'])
        (col_pts if nd['in_collision'] else free_pts).append(deg)
    node_meshes[layer_idx] = {
        'collision': pv.PolyData(np.array(col_pts,  dtype=np.float64)) if col_pts  else None,
        'free':      pv.PolyData(np.array(free_pts, dtype=np.float64)) if free_pts else None,
    }

# ── Ray mesh ──
ray_line_mesh = None
if recorded_rays:
    rpts, rcells = [], []
    offset = 0
    for r in recorded_rays:
        seg = np.degrees(np.array([r['origin'], r['end']]))
        rpts.append(seg)
        rcells.append([2, offset, offset + 1])
        offset += 2
    rpts       = np.vstack(rpts).astype(np.float64)
    rcells_arr = np.array(rcells, dtype=np.int64).ravel()
    ray_line_mesh = pv.PolyData(rpts)
    ray_line_mesh.lines = rcells_arr

# ── START→GOAL line ──
sg_line = pv.Spline(np.degrees(np.array(
    [START + t * (GOAL - START) for t in np.linspace(0, 1, 200)])), 200)

# ══════════════════════════════════════════════════════════════════
# PYVISTA INTERACTIVE PLOTTER
# ══════════════════════════════════════════════════════════════════

pl = pv.Plotter(window_size=[1500, 950])
pl.set_background("#1a1a2e")

actor_store = {}

pl.add_mesh(sg_line, color="dodgerblue", line_width=2, opacity=0.5)
pl.add_points(np.degrees(START).reshape(1,3), color="lime",   point_size=22,
              render_points_as_spheres=True)
pl.add_points(np.degrees(GOAL).reshape(1,3),  color="orange", point_size=22,
              render_points_as_spheres=True)

if red_tet_mesh is not None:
    actor_store['red_tets'] = pl.add_mesh(
        red_tet_mesh, color="#E74C3C", opacity=0.35, show_edges=False)

if green_tet_mesh is not None:
    actor_store['green_tets'] = pl.add_mesh(
        green_tet_mesh, color="#2ECC71", opacity=1, show_edges=False)

if edge_mesh is not None:
    actor_store['edges'] = pl.add_mesh(
        edge_mesh, color="#7F8C8D", line_width=1.5, opacity=0.9)

node_actors_collision = []
node_actors_free      = []
for layer_idx in range(N_LAYERS + 1):
    colour = LAYER_COLOURS[min(layer_idx, len(LAYER_COLOURS)-1)]
    nm = node_meshes[layer_idx]
    if nm['collision'] is not None:
        a = pl.add_points(nm['collision'], color=colour, point_size=14,
                          render_points_as_spheres=True)
        node_actors_collision.append(a)
    if nm['free'] is not None:
        a = pl.add_points(nm['free'], color=colour, point_size=9,
                          render_points_as_spheres=False)
        node_actors_free.append(a)

actor_store['nodes_collision'] = node_actors_collision
actor_store['nodes_free']      = node_actors_free

if ray_line_mesh is not None:
    actor_store['rays'] = pl.add_mesh(
        ray_line_mesh, color="#4a4a6a", line_width=0.8, opacity=0.4)
    actor_store['rays'].SetVisibility(False)

vis_state = {'edges': True, 'nodes': True, 'rays': False, 'red_tets': True, 'green_tets': True}

def _set_visibility(actor_or_list, val):
    if actor_or_list is None: return
    if isinstance(actor_or_list, list):
        for a in actor_or_list: a.SetVisibility(val)
    else:
        actor_or_list.SetVisibility(val)

def toggle_edges():
    vis_state['edges'] = not vis_state['edges']
    _set_visibility(actor_store.get('edges'), vis_state['edges']); pl.update()

def toggle_nodes():
    vis_state['nodes'] = not vis_state['nodes']
    _set_visibility(actor_store.get('nodes_collision'), vis_state['nodes'])
    _set_visibility(actor_store.get('nodes_free'),      vis_state['nodes']); pl.update()

def toggle_rays():
    vis_state['rays'] = not vis_state['rays']
    _set_visibility(actor_store.get('rays'), vis_state['rays']); pl.update()

def toggle_red_tets():
    vis_state['red_tets'] = not vis_state['red_tets']
    _set_visibility(actor_store.get('red_tets'), vis_state['red_tets']); pl.update()

def toggle_green_tets():
    vis_state['green_tets'] = not vis_state['green_tets']
    _set_visibility(actor_store.get('green_tets'), vis_state['green_tets']); pl.update()

pl.add_key_event('g', toggle_edges)
pl.add_key_event('n', toggle_nodes)
pl.add_key_event('r', toggle_rays)
pl.add_key_event('t', toggle_red_tets)
pl.add_key_event('y', toggle_green_tets)

pl.add_axes(xlabel="q1 (deg)", ylabel="q2 (deg)", zlabel="q3 (deg)")
pl.add_title(
    f"C-Space  |  {N_LAYERS} layers  |  {len(graph_nodes)} nodes  |  "
    f"{len(graph_edges)} edges  |  {len(shaded_tets):,} red tets\n"
    f"Keys:  [g] edges   [n] nodes   [r] rays   [t] red tets   [y] green tets",
    font_size=9, color="white")

print("\n[INFO] Opening PyVista window …")
print("  Keys:  [g] edges  [n] nodes  [r] rays  [t] red tets")
pl.show(interactive=True, auto_close=False)
total_time = time.perf_counter() - _SCRIPT_START
print(f"\n[DONE] Total time       : {total_time:.2f}s")
print(f"[DONE] Total SDF queries: {_sdf_query_count}  (unique, cache misses only)")
print("[INFO] Done.")