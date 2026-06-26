"""
cspace_3d_random_rays.py
────────────────────────
3-DOF arm, 3D C-space (q1, q2, q3).

Algorithm:
  1. Tetrahedralise C-space.
  2. Shoot N_RAYS random-direction rays from START, each of length
     RAY_LEN = |GOAL - START| in rad-space.
  3. Along each ray, sample SDF at 5 equal intervals (RAY_LEN / 5).
     Regula Falsi on first two sign-change crossings → midpoint.
     If midpoint SDF < 0 → record it as a red tet.
  4. KNN lines: connect each red tet centroid to neighbours whose
     distance < KNN_THRESH. Cap at KNN_K per node; skip if none qualify.

NOTE: No wrapping is applied anywhere. Rays travel in straight lines
from START out to RAY_LEN with no torus folding at ±π.
"""

import pybullet_data
import numpy as np
import pybullet as p
import time, os, json, glob
from collections import defaultdict
import pyvista as pv
import math

# ══════════════════════════════════════════════════════════════════
# 1.  PYBULLET SETUP
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

# ══════════════════════════════════════════════════════════════════
# 2.  LOAD OBSTACLE MESHES (objs/)
# ══════════════════════════════════════════════════════════════════

#Getting objs
hull_body_ids = []
print("[INFO] Loading obstacle meshes from objs/ ...")
for obj_path in sorted(glob.glob(os.path.join("objs", "*.obj"))):
    try:
        col_id = p.createCollisionShape(
            p.GEOM_MESH,
            fileName=obj_path,
            meshScale=[1, 1, 1],
            flags=p.GEOM_FORCE_CONCAVE_TRIMESH
        )

        body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col_id,
            basePosition=[0, 0, 0]
        )

        hull_body_ids.append(body_id)

    except Exception as e:
        print(f"Failed to load {obj_path}: {e}")

print(f"[INFO] {len(hull_body_ids)} mesh bodies ready")

# ══════════════════════════════════════════════════════════════════
# 5.  SDF
# ══════════════════════════════════════════════════════════════════

_sdf_cache       = {}
_sdf_query_count = 0

def sdf_scene(arm_id, threshold=10.0):
    min_d = threshold
    for hull_id in hull_body_ids:
        contacts = p.getClosestPoints(bodyA=arm_id, bodyB=hull_id, distance=threshold)
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
# 6.  START / GOAL
# ══════════════════════════════════════════════════════════════════

GOAL = np.array([0, 0, 0])
START  = np.array([0, math.pi/2, 0])
print(eval_sdf_vec(START))
print(eval_sdf_vec(GOAL))

# ══════════════════════════════════════════════════════════════════
# 7.  TETRAHEDRALISE C-SPACE
# ══════════════════════════════════════════════════════════════════

GRID_STEP = 0.1
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
tet_centroids = vertices[tetrahedra].mean(axis=1)
print(f"[INFO] {len(vertices):,} vertices, {M:,} tetrahedra")

# ══════════════════════════════════════════════════════════════════
# 8.  HELPERS
# ══════════════════════════════════════════════════════════════════

def find_containing_tet(cfg):
    return int(np.argmin(np.linalg.norm(tet_centroids - cfg, axis=1)))

def regula_falsi_3d(cfg_a, cfg_b, sdf_a, sdf_b, tol=1e-4, max_iter=50):
    a, b   = cfg_a.copy(), cfg_b.copy()
    fa, fb = float(sdf_a), float(sdf_b)
    for _ in range(max_iter):
        cfg_c = a + fa*(a - b)/(fb - fa)
        fc    = eval_sdf_vec(cfg_c)
        if abs(fc) < tol:
            return cfg_c, fc
        if fa*fc < 0:
            b, fb = cfg_c, fc
        else:
            a, fa = cfg_c, fc
    return cfg_c, fc

# ══════════════════════════════════════════════════════════════════
# 9.  RAY PARAMETERS
# ══════════════════════════════════════════════════════════════════

N_RAYS        = 75
RAY_LEN       = math.pow(2,0.5)*math.pi   # rad-space distance Start→Goal
RAY_STEP      = RAY_LEN / 10                     # 5 equal intervals → 6 sample points
N_RAY_SAMPLES = max(5, int(np.ceil(RAY_LEN / RAY_STEP))) + 1
t_vals        = np.linspace(0.0, RAY_LEN, N_RAY_SAMPLES)
print(f"[INFO] RAY_LEN={RAY_LEN:.4f} rad  RAY_STEP={RAY_STEP:.4f} rad  samples={N_RAY_SAMPLES}")

KNN_K      = 10      # max neighbours per node
KNN_THRESH = 200.0   # degree-space distance threshold (tune as needed)

# Uniform random directions on S² using normal-vector normalisation
rng      = np.random.default_rng(seed=42)
raw      = rng.standard_normal((N_RAYS, 3))
ray_dirs = raw / np.linalg.norm(raw, axis=1, keepdims=True)
print(f"[INFO] {N_RAYS} random rays generated")


# ══════════════════════════════════════════════════════════════════
# DRAW RAYS  (+ optional boundary crossing points)
# ══════════════════════════════════════════════════════════════════

def draw_rays(pl, start, ray_dirs, ray_len,
              n_samples=2, color="#7F8C8D", line_width=1.0,
              opacity=0.35, label="Rays",
              boundary_points=None, show_boundary_points=False,
              boundary_color="#2ECC71", boundary_point_size=10,
              single_crossing_points=None, show_single_crossing_points=False,
              single_crossing_color="#9B59B6", single_crossing_point_size=10):
    """
    Draw N_RAYS line segments from `start` out to `start + ray_len*dir`,
    in degree-space. No wrapping — rays are straight lines in rad-space,
    converted to degrees for display.

    boundary_points              : (M,3) configs (radians) from rays with
                                    2 crossings (entry+exit pairs)
    show_boundary_points         : toggle for boundary_points
    single_crossing_points       : (K,3) configs (radians) from rays with
                                    exactly 1 crossing (ray grazed the
                                    boundary but didn't exit within RAY_LEN)
    show_single_crossing_points  : toggle for single_crossing_points
    """
    t_vals_local = np.linspace(0.0, ray_len, n_samples)
    all_pts    = []
    line_cells = []
    pt_offset  = 0

    for rdir in ray_dirs:
        seg_pts = np.degrees(np.array([start + t * rdir for t in t_vals_local]))
        all_pts.append(seg_pts)
        n = len(seg_pts)
        line_cells.append(np.hstack([[n], np.arange(pt_offset, pt_offset + n)]))
        pt_offset += n

    all_pts = np.vstack(all_pts)
    lines   = np.concatenate(line_cells).astype(np.int64)

    rays_poly = pv.PolyData(all_pts)
    rays_poly.lines = lines

    pl.add_mesh(rays_poly, color=color, line_width=line_width,
                opacity=opacity, label=label)

    if show_boundary_points and boundary_points is not None and len(boundary_points) > 0:
        pts_deg = np.degrees(np.asarray(boundary_points))
        pl.add_points(pts_deg, color=boundary_color, point_size=boundary_point_size,
                      render_points_as_spheres=True, label="Ray boundary points (2x)")

    if show_single_crossing_points and single_crossing_points is not None and len(single_crossing_points) > 0:
        pts_deg = np.degrees(np.asarray(single_crossing_points))
        pl.add_points(pts_deg, color=single_crossing_color, point_size=single_crossing_point_size,
                      render_points_as_spheres=True, label="Ray boundary points (1x)")


# ══════════════════════════════════════════════════════════════════
# 10.  SCAN RAYS → MIDPOINT RED TETS
# ══════════════════════════════════════════════════════════════════

print(f"\n[INFO] Scanning {N_RAYS} rays …")
start_time = time.perf_counter()

red_tets        = []   # tet indices
mid_configs     = []   # midpoint configs (SDF < 0)
all_crossings   = []   # crossing points from rays with 2 crossings
single_crossings = []  # crossing points from rays with exactly 1 crossing

for ri, rdir in enumerate(ray_dirs):
    cfgs = [START + t * rdir for t in t_vals]
    sdfs = [eval_sdf_vec(c) for c in cfgs]

    crossings = []
    for i in range(N_RAY_SAMPLES - 1):
        if len(crossings) == 2:
            break
        if sdfs[i] * sdfs[i+1] < 0:
            cfg_z, _ = regula_falsi_3d(cfgs[i], cfgs[i+1], sdfs[i], sdfs[i+1])
            crossings.append(cfg_z)

    if len(crossings) < 2:
        if len(crossings) == 1:
            single_crossings.append(crossings[0])
        print(f"  [ray {ri:02d}] only {len(crossings)} crossing(s) — skipping")
        continue

    all_crossings.extend(crossings)

    mid_cfg = (crossings[0] + crossings[1]) / 2.0
    mid_sdf = eval_sdf_vec(mid_cfg)

    print(f"  [ray {ri:02d}]  mid=({np.degrees(mid_cfg[0]):+.1f}°,"
          f"{np.degrees(mid_cfg[1]):+.1f}°,{np.degrees(mid_cfg[2]):+.1f}°)"
          f"  SDF={mid_sdf:+.4f}")

    if mid_sdf < 0:
        ti = find_containing_tet(mid_cfg)
        red_tets.append(ti)
        mid_configs.append(mid_cfg)
        print(f"    → red tet {ti}")

end_time = time.perf_counter()

# Deduplicate
red_tets = list(set(red_tets))
print(f"\n[INFO] Scan done in {end_time - start_time:.2f}s")
print(f"  Red tets         : {len(red_tets)}")
print(f"  2-crossing rays  : {len(all_crossings)//2}")
print(f"  1-crossing rays  : {len(single_crossings)}")
print(f"  SDF queries      : {_sdf_query_count}  (cache misses only)")

# ══════════════════════════════════════════════════════════════════
# 11.  BUILD RED TET MESH
# ══════════════════════════════════════════════════════════════════

def build_tet_mesh(tet_indices):
    if len(tet_indices) == 0:
        return None
    sel            = tetrahedra[np.array(list(tet_indices), dtype=np.int64)]
    unique_vi, inv = np.unique(sel, return_inverse=True)
    local_verts    = np.degrees(vertices[unique_vi])
    local_tets     = inv.reshape(-1, 4)
    n              = len(local_tets)
    cells          = np.hstack([np.full((n,1), 4, dtype=np.int64), local_tets]).ravel()
    celltypes      = np.full(n, pv.CellType.TETRA, dtype=np.uint8)
    return pv.UnstructuredGrid(cells, celltypes, local_verts.astype(np.float64))

print("\n[INFO] Building PyVista mesh …")
red_mesh = build_tet_mesh(red_tets)

# ══════════════════════════════════════════════════════════════════
# 12.  KNN LINES  (threshold-gated, cap KNN_K per node)
# ══════════════════════════════════════════════════════════════════

knn_mesh = None
if len(red_tets) >= 2:
    # Work in degree-space (same units as KNN_THRESH)
    red_centers = np.degrees(tet_centroids[np.array(red_tets, dtype=np.int64)])

    diff  = red_centers[:, None, :] - red_centers[None, :, :]   # (R,R,3)
    dists = np.linalg.norm(diff, axis=-1)                        # (R,R)
    np.fill_diagonal(dists, np.inf)

    seen_edges = set()
    edge_pts   = []

    for i in range(len(red_tets)):
        # Candidates: within threshold
        candidates = np.where(dists[i] < KNN_THRESH)[0]
        if len(candidates) == 0:
            continue
        # Sort by distance, take up to KNN_K
        candidates = candidates[np.argsort(dists[i][candidates])][:KNN_K]
        for j in candidates:
            key = (min(i, int(j)), max(i, int(j)))
            if key not in seen_edges:
                seen_edges.add(key)
                edge_pts.append(red_centers[i])
                edge_pts.append(red_centers[int(j)])

    print(f"[INFO] {len(seen_edges)} KNN edges  (thresh={KNN_THRESH}°, k≤{KNN_K})")

    if edge_pts:
        edge_pts  = np.array(edge_pts)
        n_edges   = len(edge_pts) // 2
        cells_knn = np.hstack([
            np.full((n_edges, 1), 2, dtype=np.int64),
            np.arange(n_edges * 2, dtype=np.int64).reshape(n_edges, 2)
        ]).ravel()
        knn_mesh       = pv.PolyData(edge_pts)
        knn_mesh.lines = cells_knn

# ══════════════════════════════════════════════════════════════════
# 13.  VISUALISE
# ══════════════════════════════════════════════════════════════════

pl = pv.Plotter(window_size=[1400, 900])
pl.set_background("white")

if red_mesh is not None:
    pl.add_mesh(red_mesh, color="#C0392B", opacity=0.85,
                show_edges=True, edge_color="#7B241C", line_width=0.4,
                label=f"Collision midpoints ({len(red_tets)})")

if knn_mesh is not None:
    pl.add_mesh(knn_mesh, color="#F39C12", line_width=2.5,
                label=f"KNN edges (k≤{KNN_K}, thresh={KNN_THRESH}°)")

# START→GOAL line
pl.add_mesh(pv.Spline(np.degrees(np.array(
    [START + t*(GOAL-START) for t in np.linspace(0,1,200)]
)), 200), color="dodgerblue", line_width=4, label="START→GOAL line")

pl.add_points(np.degrees(START).reshape(1,3), color="lime",   point_size=22,
              render_points_as_spheres=True, label="Start")
pl.add_points(np.degrees(GOAL).reshape(1,3),  color="orange", point_size=22,
              render_points_as_spheres=True, label="Goal")

pl.add_axes(xlabel="q1 (deg)", ylabel="q2 (deg)", zlabel="q3 (deg)")
pl.add_legend(bcolor="white", border=True, size=(0.30, 0.18))
pl.add_title(
    f"3D C-Space | {N_RAYS} random rays | "
    f"red={len(red_tets)} | KNN k≤{KNN_K} thresh={KNN_THRESH}° | "
    f"queries={_sdf_query_count} | time={end_time-start_time:.2f}s",
    font_size=10)

SHOW_RAY_BOUNDARY_POINTS        = True   # rays with 2 crossings
SHOW_SINGLE_CROSSING_POINTS     = True   # rays with only 1 crossing

draw_rays(pl, START, ray_dirs, RAY_LEN,
          boundary_points=all_crossings,
          show_boundary_points=SHOW_RAY_BOUNDARY_POINTS,
          single_crossing_points=single_crossings,
          show_single_crossing_points=SHOW_SINGLE_CROSSING_POINTS)

print("[INFO] Opening PyVista window …")
pl.show()
p.disconnect()
print("[INFO] Done.")