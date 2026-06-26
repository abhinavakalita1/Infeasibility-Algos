"""
cspace_3d_dijkstra_pyvista_alpha.py
────────────────────────────────────
3-DOF arm, 3D C-space (q1, q2, q3).
SDF backend: alpha-shape surface reconstruction per cluster.

Algorithm Updates:
  1. Implements a strict LOSS_THRESHOLD constraint during Dijkstra neighbor pruning.
  2. Uses a Linear Support Vector Machine (SVM) classifier to cluster boundary hits and determine min_dir.
  3. Dynamic exploration bonus added to loss function based on total shaded tets.
"""

import pybullet_data
import numpy as np
from sklearn.svm import SVC
import pybullet as p
import time, os, json, heapq, glob
from collections import deque, defaultdict
import pyvista as pv

# ══════════════════════════════════════════════════════════════════
# 1.  PYBULLET SETUP
# ══════════════════════════════════════════════════════════════════

physicsClient = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
planeId = p.loadURDF("plane.urdf")

p.configureDebugVisualizer(p.COV_ENABLE_GUI,            0)
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS,        1)
p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
p.resetDebugVisualizerCamera(
    cameraDistance=5.0, cameraYaw=45, cameraPitch=-30,
    cameraTargetPosition=[0, 0, 1])

arm3Id = p.loadURDF(
    "arm_3.urdf", basePosition=[0, 0, 0],
    baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
    useFixedBase=True,
    flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION)
NUM_JOINTS = p.getNumJoints(arm3Id)
print(f"[INFO] 3-DOF arm loaded — {NUM_JOINTS} joints")

# ══════════════════════════════════════════════════════════════════
# 2.  LOAD OBSTACLE MESHES (objs/)
# ══════════════════════════════════════════════════════════════════

#Getting objs
recon_body_ids  = []
recon_obj_paths = []

t_recon_start = time.perf_counter()
print("[INFO] Loading obstacle meshes from objs/ ...")
for obj_path in sorted(glob.glob(os.path.join("objs", "*.obj"))):
    try:
        col_id = p.createCollisionShape(
            p.GEOM_MESH,
            fileName=obj_path,
            meshScale=[1, 1, 1],
            flags=p.GEOM_FORCE_CONCAVE_TRIMESH
        )
        vis_id = p.createVisualShape(
            p.GEOM_MESH, fileName=obj_path,
            meshScale=[1, 1, 1],
            rgbaColor=[0.4, 0.7, 1.0, 0.6]
        )

        body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=[0, 0, 0]
        )

        recon_body_ids.append(body_id)
        recon_obj_paths.append(obj_path)
        print(f"  Loaded {obj_path} → body_id={body_id}")

    except Exception as e:
        print(f"Failed to load {obj_path}: {e}")

t_recon = (time.perf_counter() - t_recon_start) * 1000
print(f"\n[INFO] Mesh loading done in {t_recon:.1f} ms")
print(f"[INFO] {len(recon_body_ids)} mesh bodies ready")
if len(recon_body_ids) == 0:
    raise RuntimeError(
        "No mesh bodies were loaded — check that objs/ contains .obj files."
    )

# ══════════════════════════════════════════════════════════════════
# 6.  SDF
# ══════════════════════════════════════════════════════════════════

_sdf_cache       = {}
_sdf_query_count = 0

def sdf_scene(arm_id, threshold=10.0):
    min_d = threshold
    for body_id in recon_body_ids:
        contacts = p.getClosestPoints(bodyA=arm_id, bodyB=body_id, distance=threshold)
        if contacts:
            d = min(c[8] for c in contacts)
            if d < min_d: min_d = d
    return min_d

def set_config(q1, q2, q3):
    p.resetJointState(arm3Id, 0, float(q1))
    p.resetJointState(arm3Id, 1, float(q2))
    p.resetJointState(arm3Id, 2, float(q3))
    p.stepSimulation()

def eval_sdf(q1, q2, q3):
    global _sdf_query_count
    key = (round(float(q1),5), round(float(q2),5), round(float(q3),5))
    if key in _sdf_cache:
        return _sdf_cache[key]
    set_config(q1, q2, q3)
    d = sdf_scene(arm3Id)
    _sdf_cache[key] = d
    _sdf_query_count += 1
    return d

def eval_sdf_vec(cfg):
    return eval_sdf(cfg[0], cfg[1], cfg[2])

# ══════════════════════════════════════════════════════════════════
# 7.  START / GOAL
# ══════════════════════════════════════════════════════════════════
import math
START = np.array([0.0, 0.0, 0.0])
GOAL  = np.array([0.0, math.pi, math.pi])
print(eval_sdf_vec(START))
print(eval_sdf_vec(GOAL))

# ══════════════════════════════════════════════════════════════════
# 8.  LINE SCAN
# ══════════════════════════════════════════════════════════════════

line_length = np.linalg.norm(GOAL - START)
N_SEGMENTS  = max(5, int(np.ceil(line_length / 0.1)))
N_SAMPLES   = N_SEGMENTS + 1
t_vals      = np.linspace(0.0, 1.0, N_SAMPLES)
sample_cfgs = np.array([START + t*(GOAL-START) for t in t_vals])

print(f"\n[INFO] Line scan: {N_SAMPLES} samples …")
sdf_vals = []
for idx, cfg in enumerate(sample_cfgs):
    d = eval_sdf_vec(cfg)
    if idx == 0 or idx == N_SAMPLES-1:
        d = abs(d) if d != 0 else 1e-6
    sdf_vals.append(d)
    print(f"  [{idx:3d}] t={t_vals[idx]:.2f}  "
          f"q=({np.degrees(cfg[0]):+.1f}°,{np.degrees(cfg[1]):+.1f}°,{np.degrees(cfg[2]):+.1f}°)  "
          f"SDF={d:+.4f} ({'COLL' if d<0 else 'free'})")
sdf_vals = np.array(sdf_vals)

# ══════════════════════════════════════════════════════════════════
# 9.  REGULA FALSI
# ══════════════════════════════════════════════════════════════════

def regula_falsi_3d(cfg_a, cfg_b, sdf_a, sdf_b, tol=1e-4, max_iter=50):
    a, b = cfg_a.copy(), cfg_b.copy()
    fa, fb = sdf_a, sdf_b
    for _ in range(max_iter):
        cfg_c = a + fa*(a-b)/(fb-fa)
        fc    = eval_sdf_vec(cfg_c)
        if abs(fc) < tol: return cfg_c, fc
        if fa*fc < 0: b, fb = cfg_c, fc
        else:          a, fa = cfg_c, fc
    return cfg_c, fc

rf_zeros = []
print(f"\n[INFO] Regula Falsi …")
for i in range(N_SEGMENTS):
    si, sj = sdf_vals[i], sdf_vals[i+1]
    if si*sj < 0:
        cfg_r, sdf_r = regula_falsi_3d(sample_cfgs[i], sample_cfgs[i+1], si, sj)
        rf_zeros.append(cfg_r)
        print(f"  Seg {i}→{i+1}: zero at "
              f"({np.degrees(cfg_r[0]):+.2f}°,{np.degrees(cfg_r[1]):+.2f}°,{np.degrees(cfg_r[2]):+.2f}°)  "
              f"SDF={sdf_r:+.5f}")
print(f"  Total RF zeros: {len(rf_zeros)}")

# ══════════════════════════════════════════════════════════════════
# 10.  TETRAHEDRALISE C-SPACE
# ══════════════════════════════════════════════════════════════════

GRID_STEP = 0.35
q_vals = np.arange(-np.pi, np.pi + GRID_STEP*0.5, GRID_STEP)
N = len(q_vals)
print(f"\n[INFO] Grid: {N}x{N}x{N} nodes (step={GRID_STEP:.2f} rad)")

def vidx(i,j,k): return i*N*N + j*N + k

vertices = np.array([[q_vals[i], q_vals[j], q_vals[k]]
                     for i in range(N) for j in range(N) for k in range(N)])

def cube_tets(i, j, k):
    v = [vidx(i+di, j+dj, k+dk) for di,dj,dk in [
        (0,0,0),(1,0,0),(0,1,0),(1,1,0),
        (0,0,1),(1,0,1),(0,1,1),(1,1,1)
    ]]
    if (i+j+k) % 2 == 0:
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
    for tet in cube_tets(i,j,k)
], dtype=np.int32)

M             = len(tetrahedra)
tri_centroids = vertices[tetrahedra].mean(axis=1)
print(f"[INFO] {len(vertices):,} vertices, {M:,} tetrahedra")

# ══════════════════════════════════════════════════════════════════
# 11.  ADJACENCY
# ══════════════════════════════════════════════════════════════════

print("[INFO] Building adjacency …")
face_to_tets = defaultdict(list)
for ti, tet in enumerate(tetrahedra):
    for fi in range(4):
        face = tuple(sorted(tet[j] for j in range(4) if j != fi))
        face_to_tets[face].append(ti)

neighbours = [[] for _ in range(M)]
for face, tis in face_to_tets.items():
    if len(tis) == 2:
        a, b = tis
        neighbours[a].append(b)
        neighbours[b].append(a)
print("[INFO] Adjacency built.")

# ══════════════════════════════════════════════════════════════════
# 12.  FIND CONTAINING TET
# ══════════════════════════════════════════════════════════════════

def find_containing_tet(cfg):
    return int(np.argmin(np.linalg.norm(tri_centroids - cfg, axis=1)))

# ══════════════════════════════════════════════════════════════════
# 13.  MIDPOINTS → SEED TETS
# ══════════════════════════════════════════════════════════════════

start_processing_time = time.perf_counter()

seed_tets  = []
midpoints  = []
first_valid_midpoint = None   # used to anchor the 14-ray sweep

print(f"\n[INFO] Computing midpoints for consecutive RF zero pairs …")
for k in range(0, len(rf_zeros)-1, 2):
    mid     = (rf_zeros[k] + rf_zeros[k+1]) / 2.0
    mid_sdf = eval_sdf_vec(mid)
    midpoints.append((mid, mid_sdf))
    print(f"  Pair ({k},{k+1}) midpoint SDF={mid_sdf:+.5f} ({'COLL' if mid_sdf<0 else 'free'})")
    if mid_sdf < 0:
        ti = find_containing_tet(mid)
        seed_tets.append(ti)
        print(f"    → seed tet {ti}")
        if first_valid_midpoint is None:
            first_valid_midpoint = mid

if not seed_tets and rf_zeros:
    ti = find_containing_tet(rf_zeros[0])
    if eval_sdf_vec(tri_centroids[ti]) < 0:
        seed_tets.append(ti)
        first_valid_midpoint = tri_centroids[ti]
        print(f"[INFO] Fallback seed from first RF zero → tet {ti}")

if not seed_tets:
    print("[WARN] No seed tet from RF zeros — scanning …")
    for ti in range(0, M, max(1, M//500)):
        if eval_sdf_vec(tri_centroids[ti]) < 0:
            seed_tets.append(ti)
            first_valid_midpoint = tri_centroids[ti]
            print(f"  Found fallback seed tet {ti}")
            break

print(f"\n[INFO] Seed tets: {seed_tets}")

# ══════════════════════════════════════════════════════════════════
# 14.  14-RAY BOUNDARY SWEEP WITH SVM → min_dir
# ══════════════════════════════════════════════════════════════════

RAY_STEP    = 0.05   # rad — march step along each ray
RAY_MAX_LEN = 2.5    # rad — max ray length before giving up

# 6 axial + 8 diagonal directions
_ax = np.eye(3)
_diag_signs = np.array([[s0,s1,s2]
                         for s0 in [1,-1]
                         for s1 in [1,-1]
                         for s2 in [1,-1]], dtype=float)
RAY_DIRS = np.vstack([_ax, -_ax, _diag_signs / np.sqrt(3)])  # shape (14, 3)

def find_boundary_on_ray(origin, direction):
    cfg_prev  = origin.copy()
    sdf_prev  = eval_sdf_vec(cfg_prev)
    dist = 0.0

    while dist < RAY_MAX_LEN:
        dist    += RAY_STEP
        cfg_cur  = origin + dist * direction
        cfg_cur  = np.arctan2(np.sin(cfg_cur), np.cos(cfg_cur))
        sdf_cur  = eval_sdf_vec(cfg_cur)

        if sdf_prev < 0 < sdf_cur:
            bnd, _ = regula_falsi_3d(cfg_prev, cfg_cur, sdf_prev, sdf_cur)
            return bnd

        cfg_prev = cfg_cur
        sdf_prev = sdf_cur

    return None


def compute_min_dir(anchor_cfg):
    """
    Shoot 14 rays, collect boundary hits, cluster them into 2 groups using SVM,
    and find the principal direction (thin axis) of cluster 1.
    """
    print("\n[INFO] 14-ray boundary sweep …")
    boundary_hits = []
    for di, direction in enumerate(RAY_DIRS):
        bnd = find_boundary_on_ray(anchor_cfg, direction)
        if bnd is not None:
            boundary_hits.append(bnd)
            label = (f"+e{di}" if di < 3
                     else f"-e{di-3}" if di < 6
                     else f"diag{di-6}")
            print(f"  Ray {di:2d} ({label}): hit "
                  f"({np.degrees(bnd[0]):+.2f}°, {np.degrees(bnd[1]):+.2f}°, "
                  f"{np.degrees(bnd[2]):+.2f}°)")
        else:
            print(f"  Ray {di:2d}: no hit within {RAY_MAX_LEN} rad")

    if len(boundary_hits) < 4:
        print("[WARN] Too few boundary hits for robust SVM separation — defaulting min_dir to q3")
        return np.array([0.0, 0.0, 1.0]), 2, boundary_hits, np.zeros(len(boundary_hits), int)

    hits_arr = np.array(boundary_hits)   # (H, 3)

    # ── Initial binary labels using distance to anchor (midpoint) to train SVM ──
    mid_dists = np.linalg.norm(hits_arr - anchor_cfg, axis=1)
    median_dist = np.median(mid_dists)
    initial_labels = np.where(mid_dists <= median_dist, 0, 1)

    if len(np.unique(initial_labels)) < 2:
        initial_labels[0] = 1 # Force separation if points are completely uniform

    # ── SVM Classification Boundary ──────────────────────────────
    clf = SVC(kernel='linear', C=1.0, random_state=42)
    clf.fit(hits_arr, initial_labels)
    cluster_labels = clf.predict(hits_arr)

    c1_pts = hits_arr[cluster_labels == 0]
    if len(c1_pts) < 2:
        c1_pts = hits_arr # Fallback to all hits if subset is too small

    # Compute coordinate spreads/spans to locate thinnest axis
    axes = ["q1", "q2", "q3"]
    spans = []
    for ax in range(3):
        col = c1_pts[:, ax]
        span = col.max() - col.min()
        spans.append(span)
        print(f"    {axes[ax]}: max={np.degrees(col.max()):+.2f}°  "
              f"min={np.degrees(col.min()):+.2f}°  "
              f"span={np.degrees(span):.2f}°")

    axis_idx = int(np.argmin(spans))
    min_dir_unit = np.zeros(3)
    min_dir_unit[axis_idx] = 1.0

    print(f"\n[INFO] SVM clustering complete. min_dir axis = {axes[axis_idx]}  "
          f"(span={np.degrees(spans[axis_idx]):.2f}°) → "
          f"unit={min_dir_unit}")

    return min_dir_unit, axis_idx, boundary_hits, cluster_labels


# Run the sweep if valid midpoint exists
if first_valid_midpoint is not None:
    min_dir_unit, min_dir_axis_idx, boundary_hits, boundary_cluster_labels = \
        compute_min_dir(first_valid_midpoint)
else:
    print("[WARN] No valid midpoint for 14-ray sweep — defaulting min_dir to q3")
    min_dir_unit             = np.array([0.0, 0.0, 1.0])
    min_dir_axis_idx         = 2
    boundary_hits            = []
    boundary_cluster_labels  = np.array([], dtype=int)

# ══════════════════════════════════════════════════════════════════
# 15.  WEIGHTS & CRITICAL EXPANSION PRUNING THRESHOLD
# ══════════════════════════════════════════════════════════════════
W_SDF = 1.0
W_DIR = 5.0

# ── NEW PARAMETER: PRUNING THRESHOLD ──────────────────────────────
# Sets a ceiling limit on acceptable loss. Elements scoring above
# this threshold will immediately be discarded from exploration.
LOSS_THRESHOLD = 0.1  # Adjust lower to restrict search, higher to open exploration

def compute_loss(parent_centroid, child_centroid, child_sdf, min_dir, num_shaded):
    sdf_term = W_SDF * child_sdf
    delta    = child_centroid - parent_centroid
    dir_term = W_DIR * abs(np.dot(delta, min_dir))

    # Adaptive exploration bonus term
    bonus_term = -(0.05 - min(0.05, num_shaded / 1000.0))

    return sdf_term + dir_term + bonus_term


def update_min_dir(parent_centroid, child_centroid, min_dir):
    delta = child_centroid - parent_centroid
    delta_len = np.linalg.norm(delta)
    if delta_len < 1e-9:
        return min_dir

    delta_unit = delta / delta_len
    v_perp = delta_unit - np.dot(delta_unit, min_dir) * min_dir
    v_perp_len = np.linalg.norm(v_perp)
    if v_perp_len < 1e-9:
        return min_dir

    v_perp_unit = v_perp / v_perp_len
    candidate_pos = min_dir + v_perp_unit
    candidate_neg = min_dir - v_perp_unit

    if np.linalg.norm(candidate_pos) < np.linalg.norm(candidate_neg):
        blended = candidate_pos
    else:
        blended = candidate_neg

    blended_len = np.linalg.norm(blended)
    if blended_len < 1e-9:
        return min_dir

    return blended / blended_len

# ══════════════════════════════════════════════════════════════════
# 16.  SEPARATION CHECK
# ══════════════════════════════════════════════════════════════════

def is_separated(shaded_set):
    start_ti = find_containing_tet(START)
    goal_ti  = find_containing_tet(GOAL)
    visited  = set()
    queue    = deque([start_ti])
    while queue:
        ti = queue.popleft()
        if ti in visited: continue
        if ti == goal_ti: return False
        visited.add(ti)
        for nb in neighbours[ti]:
            if nb not in visited and nb not in shaded_set:
                queue.append(nb)
    return True

# ══════════════════════════════════════════════════════════════════
# 17.  DIJKSTRA — WITH CRITICAL EDGE LOSS THRESHOLDING
# ══════════════════════════════════════════════════════════════════

CHECK_EVERY = 20

shaded   = set()
red_tets = []
stop_reasons = []

current_min_dir = min_dir_unit.copy()

if seed_tets:
    for pair_idx, seed_ti in enumerate(seed_tets):
        if seed_ti in shaded:
            stop_reasons.append(f"pair {pair_idx}: seed already shaded")
            continue

        seed_sdf = eval_sdf_vec(tri_centroids[seed_ti])
        shaded.add(seed_ti)
        red_tets.append(seed_ti)

        # Heap entries: (loss, child_ti, parent_centroid)
        heap = []
        for nb in neighbours[seed_ti]:
            if nb not in shaded:
                nb_sdf = eval_sdf_vec(tri_centroids[nb])
                if nb_sdf < 0:
                    loss = compute_loss(tri_centroids[seed_ti],
                                        tri_centroids[nb],
                                        nb_sdf,
                                        current_min_dir,
                                        len(shaded))
                    # Early Loss Threshold Verification
                    if loss <= LOSS_THRESHOLD:
                        heapq.heappush(heap, (loss, nb, seed_ti))

        dead = set()
        step = 0
        pair_stop_reason = "heap exhausted"

        print(f"\n[INFO] Dijkstra from seed tet {seed_ti}  (SDF={seed_sdf:+.5f})")
        print(f"  Initial min_dir = {current_min_dir}  "
              f"(w_sdf={W_SDF}, w_dir={W_DIR}, Max Allowed Loss={LOSS_THRESHOLD})")

        while heap:
            loss, ti, parent_ti = heapq.heappop(heap)
            if ti in shaded or ti in dead:
                continue

            sdf_val = eval_sdf_vec(tri_centroids[ti])
            if sdf_val >= 0:
                dead.add(ti)
                continue

            # Shade this tet
            parent_c = tri_centroids[parent_ti]
            child_c  = tri_centroids[ti]

            shaded.add(ti)
            red_tets.append(ti)
            step += 1

            # Update direction matrix tracking
            current_min_dir = update_min_dir(parent_c, child_c, current_min_dir)

            # Separation verification check
            separated = False
            if step % CHECK_EVERY == 0:
                separated = is_separated(shaded)

            if step % 50 == 0 or separated:
                print(f"  step {step:6d}  tet {ti:6d}  SDF={sdf_val:+.5f}  "
                      f"loss={loss:+.5f}  shaded={len(shaded)}  "
                      f"min_dir=[{current_min_dir[0]:+.3f},{current_min_dir[1]:+.3f},"
                      f"{current_min_dir[2]:+.3f}]  "
                      f"{'SEPARATED' if separated else ''}")

            if separated:
                pair_stop_reason = f"separated after {len(shaded)} tets"
                break

            # Process adjacent elements
            for nb in neighbours[ti]:
                if nb not in shaded and nb not in dead:
                    nb_sdf = eval_sdf_vec(tri_centroids[nb])
                    if nb_sdf < 0:
                        nb_loss = compute_loss(child_c,
                                               tri_centroids[nb],
                                               nb_sdf,
                                               current_min_dir,
                                               len(shaded))
                        # PRUNING: Only push to priority queue if it passes threshold criteria
                        if nb_loss <= LOSS_THRESHOLD:
                            heapq.heappush(heap, (nb_loss, nb, ti))

        stop_reasons.append(f"pair {pair_idx}: {pair_stop_reason}")
        if "separated" in pair_stop_reason:
            break

    _stop_reason = "; ".join(stop_reasons)
else:
    _stop_reason = "no valid seed tetrahedra found"

end_processing_time = time.perf_counter()
total_time = end_processing_time - start_processing_time

print(f"\n[INFO] Dijkstra done.")
print(f"  Stop reason : {_stop_reason}")
print(f"  Shaded tets : {len(red_tets)}")
print(f"  Time        : {total_time:.4f}s")
print(f"  SDF queries : {_sdf_query_count}")
print(f"  Final min_dir: {current_min_dir}")

# ══════════════════════════════════════════════════════════════════
# 18.  BUILD PYVISTA MESH
# ══════════════════════════════════════════════════════════════════

def build_tet_mesh(tet_indices, verts_rad):
    if len(tet_indices) == 0:
        return None
    sel_tets       = tetrahedra[np.array(tet_indices, dtype=np.int64)]
    unique_vi, inv = np.unique(sel_tets, return_inverse=True)
    local_verts    = np.degrees(verts_rad[unique_vi])
    local_tets     = inv.reshape(-1, 4)
    n_tets         = len(local_tets)
    cells          = np.hstack([np.full((n_tets,1), 4, dtype=np.int64), local_tets]).ravel()
    celltypes      = np.full(n_tets, pv.CellType.TETRA, dtype=np.uint8)
    return pv.UnstructuredGrid(cells, celltypes, local_verts.astype(np.float64))

print("\n[INFO] Building PyVista mesh …")
red_mesh = build_tet_mesh(red_tets, vertices)

# ══════════════════════════════════════════════════════════════════
# 19.  PLOT
# ══════════════════════════════════════════════════════════════════

pl = pv.Plotter(window_size=[1400, 900])
pl.set_background("white")

if red_mesh is not None:
    pl.add_mesh(red_mesh, color="#C0392B", opacity=0.85,
                show_edges=True, edge_color="#7B241C", line_width=0.4,
                label=f"Collision tets ({len(red_tets)})")

line_pts = np.degrees(sample_cfgs)
spline   = pv.Spline(line_pts, 200)
pl.add_mesh(spline, color="dodgerblue", line_width=4, label="C-space path")

if rf_zeros:
    pl.add_points(np.degrees(np.array(rf_zeros)), color="purple", point_size=18,
                  render_points_as_spheres=True, label="RF zeros")

if midpoints:
    pl.add_points(np.degrees(np.array([m for m,_ in midpoints])), color="yellow",
                  point_size=16, render_points_as_spheres=True, label="Midpoints")

if boundary_hits:
    hits_deg = np.degrees(np.array(boundary_hits))
    cluster_colors = ["cyan", "magenta"]
    n_clusters_plot = int(boundary_cluster_labels.max()) + 1 if len(boundary_cluster_labels) else 0
    for ci in range(n_clusters_plot):
        mask = boundary_cluster_labels == ci
        if mask.any():
            pl.add_points(hits_deg[mask],
                          color=cluster_colors[ci % len(cluster_colors)],
                          point_size=14,
                          render_points_as_spheres=True,
                          label=f"SVM Class {ci}")

if first_valid_midpoint is not None:
    origin_deg = np.degrees(first_valid_midpoint)
    arrow_len  = 20.0
    arrow_tip  = origin_deg + arrow_len * min_dir_unit * (180.0 / np.pi)
    arrow = pv.Arrow(start=origin_deg,
                     direction=min_dir_unit,
                     scale=arrow_len,
                     tip_length=0.3, tip_radius=0.1, shaft_radius=0.03)
    pl.add_mesh(arrow, color="gold", label=f"min_dir ({['q1','q2','q3'][min_dir_axis_idx]})")

pl.add_points(np.degrees(START).reshape(1,3), color="lime",   point_size=22,
              render_points_as_spheres=True, label="Start")
pl.add_points(np.degrees(GOAL).reshape(1,3),  color="orange", point_size=22,
              render_points_as_spheres=True, label="Goal")

pl.add_axes(xlabel="q1 (deg)", ylabel="q2 (deg)", zlabel="q3 (deg)")
pl.add_legend(bcolor="white", border=True, size=(0.30, 0.26))
pl.add_title(
    f"3D C-Space | alpha-recon α= | {len(red_tets)} tets | "
    f"min_dir={['q1','q2','q3'][min_dir_axis_idx]} | "
    f"Threshold Max={LOSS_THRESHOLD} | time={total_time:.2f}s",
    font_size=9)

print("[INFO] Opening PyVista window …")
pl.show()
p.disconnect()
print("[INFO] Done.")