"""
cspace_3d_frontier_growth.py
─────────────────────────────
3-DOF arm, 3D C-space (q1, q2, q3).

Algorithm:
  1. Tetrahedralise C-space (unchanged from ray-cast version).
  2. Walk the straight line START→GOAL, sampling SDF at intervals.
     Regula Falsi every sign change → boundary points on the line.
     Pair consecutive boundary points, keep midpoints with SDF < 0,
     shade the enclosing tet(s) red. The midpoint of the *last* such
     pair becomes the ROOT seed point ("P0") for frontier growth.
  3. Round 1 (root → 4 named branch tips):
       - From ROOT, scatter N_RAYS=8 random directions. Walk each ray
         in steps; stop at the FIRST sign change found (regula falsi)
         → at most one boundary point per ray. Rays that never cross
         the SDF=0 surface within RAY_LEN contribute nothing (skipped).
       - Take all pairwise combinations of the boundary points found
         this round. Compute each pair's midpoint; keep only midpoints
         with SDF < 0 (strictly negative).
       - Rank surviving midpoints by Euclidean (rad-space) distance
         from ROOT; keep the 4 farthest (fewer if <4 exist).
       - Shade + NAME each kept midpoint's tet ("P1".."P4"). These
         become the 4 active branch tips. Discard every other
         candidate midpoint and every boundary point not used by a
         kept pair — none of that gets stored or plotted.
  4. Round 2+ (4 parallel branches, forever):
       - For EACH of the 4 current branch tips independently:
           • scatter N_RAYS=8 rays from that tip (first-crossing only)
           • pairwise-combine the boundary points found
           • keep midpoints with SDF < 0
           • pick the ONE candidate farthest (Euclidean, rad-space)
             from THIS branch's own tip
           • shade + name it; it replaces this branch's tip for the
             next round
           • ADDITIONALLY: walk the tet mesh from the parent's tet
             toward the child along the parent→child direction (see
             step 4b below), shading a corridor of red tets between
             them. This corridor is never named/registered as a point
             and never spawns its own rays — it's pure shading.
       - Each round therefore adds exactly 4 new shaded/named tets
         (one per branch), plus however many corridor tets the walk
         shaded along the way. Discarded candidates/boundary points
         are never stored or plotted.
  4b. PARENT→CHILD TET WALK (corridor shading):
       - Start at the tet containing the parent point.
       - Repeatedly step to the face-neighbour tet that makes forward
         progress along the parent→child direction (measured from the
         start tet) AND has the smallest perpendicular distance to the
         straight parent→child line — this keeps the walk hugging the
         line instead of drifting/zig-zagging through the 3D mesh.
       - At each stepped-to tet, check if ALL 4 of its vertices have
         SDF < 0 ("complete -ve sdf simplex"). If yes, shade it red
         and continue walking from there.
       - If NO — i.e. the one neighbour lying in the parent→child
         direction has SDF >= 0 on some vertex — this is a BOUNDARY
         TOUCH, see step 4c below.
       - The walk also stops cleanly (no detour) if it reaches the
         child's own tet, runs out of neighbours, or covers the full
         parent→child distance, whichever comes first.
       - No line/segment is plotted for this corridor — only the
         shaded tets themselves are visualised, via the same red
         tet mesh as everything else.
  4c. BOUNDARY TOUCH → DETOUR (new):
       - When the corridor walk's one directional neighbour has
         SDF >= 0, the walk does not just stop — it spawns a detour:
           1. N_PTS_TARGET is incremented by 1 (a new midpoint is
              being added to the growth budget).
           2. The tet where the walk stopped is itself registered as
              a new named point, parented to whatever named point
              started this corridor leg. This draws the
              parent→current-tet portion of the corridor.
           3. From that new point, N_RAYS=8 rays are scattered exactly
              like a normal growth round; the farthest negative-SDF
              pairwise midpoint is kept and registered as ANOTHER new
              named point, parented to the current-tet point.
           4. The journey continues: a fresh corridor walk runs from
              that new midpoint toward the ORIGINAL child — which may
              itself hit another boundary touch and recurse again.
       - Net effect: an edge that would have been a single
         parent→child corridor becomes
             parent → current_tet → new_midpoint → ... → child
         i.e. every boundary touch detours around the obstacle via a
         freshly-scattered midpoint before continuing toward the
         original destination, rather than the corridor simply
         stopping short.
  5. Repeat step 4 until N_PTS_TARGET MIDPOINTS have been chosen via
     scattering (i.e. len(node_names) reaches the target — note
     N_PTS_TARGET itself grows by +1 per boundary-touch detour, see
     4c, so the budget expands to account for detour points; corridor
     filler tets that are NOT registered as named points still do not
     count toward this), or a branch stalls (its round produces no
     valid candidate — that branch is simply skipped that round,
     others keep going).

VISUALISATION: only the KEPT points (P0, P1..P4, every named
descendant, and every detour point spawned by a boundary touch), the
8 rays that were scattered from each kept point, and the parent→child
tet-walk corridors (including detour legs) are drawn. The cloud of
discarded candidate midpoints and unused boundary points is never
plotted. No explicit parent→child connector LINES are ever drawn —
the chain of shaded red tets is itself the only visual representation
of a parent→child (or detour) connection.

NOTE: No torus wrapping is applied anywhere — all rays/lines travel in
straight rad-space lines with no folding at ±π.
"""

import pybullet_data
import numpy as np
import pybullet as p
import time, os, json, glob, itertools
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

GOAL  = np.array([0, 0, 0])
START = np.array([0, math.pi/2, 0])
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
# 7b.  TET FACE-ADJACENCY (for parent→child corridor walk)
# ══════════════════════════════════════════════════════════════════
# Two tets are neighbours if they share a face (3 of their 4 vertices).
# Build once: map each face (sorted vertex-id triple) -> list of tet
# indices that contain it. Then each tet's neighbours are, for each of
# its 4 faces, the *other* tet sharing that face (if any — boundary
# faces have only one owner).

print("[INFO] Building tet face-adjacency …")
_face_to_tets = defaultdict(list)
_tet_faces    = np.empty((M, 4), dtype=object)  # tet_idx -> 4 face keys

for ti in range(M):
    a, b, c, d = tetrahedra[ti]
    faces = (
        tuple(sorted((a, b, c))),
        tuple(sorted((a, b, d))),
        tuple(sorted((a, c, d))),
        tuple(sorted((b, c, d))),
    )
    _tet_faces[ti] = faces
    for f in faces:
        _face_to_tets[f].append(ti)

tet_neighbors = [[] for _ in range(M)]
for ti in range(M):
    for f in _tet_faces[ti]:
        for tj in _face_to_tets[f]:
            if tj != ti:
                tet_neighbors[ti].append(tj)

print(f"[INFO] Adjacency built — avg neighbours/tet = "
      f"{np.mean([len(n) for n in tet_neighbors]):.2f}")

# ══════════════════════════════════════════════════════════════════
# 8.  HELPERS
# ══════════════════════════════════════════════════════════════════

def find_containing_tet(cfg):
    return int(np.argmin(np.linalg.norm(tet_centroids - cfg, axis=1)))

def regula_falsi_3d(cfg_a, cfg_b, sdf_a, sdf_b, tol=1e-4, max_iter=50):
    a, b   = cfg_a.copy(), cfg_b.copy()
    fa, fb = float(sdf_a), float(sdf_b)
    cfg_c, fc = a.copy(), fa
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

def tet_all_negative(ti):
    """True iff ALL 4 vertices of tet ti have SDF < 0 ('complete -ve sdf simplex')."""
    for vi in tetrahedra[ti]:
        cfg = vertices[vi]
        if eval_sdf_vec(cfg) >= 0:
            return False
    return True

def best_directional_neighbour(cur_ti, start_centroid, direction, cur_proj, max_proj, visited):
    """
    Shared neighbour-selection logic used by the corridor walk: among
    cur_ti's unvisited face-neighbours, keep only those that make
    strict forward progress along `direction` (measured from
    start_centroid) beyond cur_proj. Two tiers:

      TIER 1 (preferred): neighbours that also stay within max_proj
      (don't overshoot the child/target distance). Among these, return
      the one with the smallest perpendicular distance to the straight
      line through start_centroid along `direction` (keeps the walk
      hugging the line).

      TIER 2 (fallback): if NO neighbour clears tier 1 — this happens
      whenever target_dist is smaller than the local tet spacing, e.g.
      two scattered midpoints that land only ~0.02 rad apart while
      adjacent tet centroids are ~0.05 rad apart, so literally every
      forward step "overshoots" the 5%-slack cap — fall back to the
      forward-progressing neighbour with the SMALLEST OVERSHOOT past
      max_proj (closest landing to the child), still tie-broken by
      perpendicular distance. Without this fallback the walk silently
      returns an empty corridor any time start/goal are closer together
      than one tet hop, even though a perfectly good single step exists
      — that's what caused points like nearby scattered siblings to
      show up with zero shaded tets connecting them.

    Returns (best_ti, best_proj, is_overshoot). is_overshoot is True
    iff the pick came from tier 2 — the CALLER must treat that as the
    walk's last allowed step (see walk_corridor): chaining multiple
    tier-2 picks in a row has no distance limit at all and will run
    the corridor arbitrarily far past the child, since once cur_proj
    exceeds max_proj every future step is "overshoot" by definition.
    (best_ti, None, False) only if truly no neighbour makes any
    forward progress at all (genuine dead end).
    """
    neighbours = [nb for nb in tet_neighbors[cur_ti] if nb not in visited]

    best_ti, best_perp, best_proj = None, np.inf, None             # tier 1
    fb_ti, fb_overshoot, fb_proj   = None, np.inf, None             # tier 2

    for nb in neighbours:
        offset = tet_centroids[nb] - start_centroid
        proj   = np.dot(offset, direction)
        if proj <= cur_proj:
            continue   # not forward progress — reject outright

        perp_vec = offset - proj * direction
        perp     = np.linalg.norm(perp_vec)

        if proj <= max_proj:
            if perp < best_perp:
                best_perp, best_ti, best_proj = perp, nb, proj
        else:
            overshoot = proj - max_proj
            if overshoot < fb_overshoot:
                fb_overshoot, fb_ti, fb_proj = overshoot, nb, proj

    if best_ti is not None:
        return best_ti, best_proj, False
    return fb_ti, fb_proj, (fb_ti is not None)

def walk_corridor(start_ti, goal_ti, direction, target_dist, max_steps=500):
    """
    Walk the tet mesh from start_ti toward goal_ti, hugging the straight
    line start->goal as closely as possible (see best_directional_neighbour
    for the line-hugging neighbour-selection rule, which fixes the
    earlier zig-zag/ribbon problem: face-adjacency in a tet mesh offers
    several "forward-ish" neighbours at every step, so we must pick the
    one closest to the true line, not the one that jumps farthest).

    The walk stops once its forward progress reaches ~target_dist (the
    parent->child distance, with a little slack), at a dead end, after
    max_steps, or — IMPORTANT — the moment the single neighbour that
    lies ALONG `direction` from the current tet has SDF >= 0 on any of
    its vertices ("boundary touch"). In that case the walk does NOT
    silently stop and discard progress: it reports the touch via
    `boundary_event` so the caller (register_point) can spawn a detour
    midpoint right there and keep going toward the original goal.

    Returns (corridor, boundary_event):
      corridor       : list of tet indices shaded along the way (NOT
                        including start_ti itself). Always non-empty
                        for any start_ti != goal_ti with at least one
                        forward-progressing neighbour, even when
                        target_dist is smaller than one tet hop apart
                        (tier-2 fallback in best_directional_neighbour
                        guarantees at least one step is taken in that
                        case, then the walk stops — see is_overshoot).
      boundary_event : None if the walk reached goal_ti / target_dist /
                        a dead end / a tier-2 fallback step cleanly,
                        otherwise the tet index where the walk stopped
                        because ITS directional neighbour was >= 0 SDF
                        (i.e. corridor[-1] if corridor else start_ti —
                        the "current tet" the detour should scatter
                        rays from).
    """
    corridor       = []
    visited        = {start_ti}
    cur_ti         = start_ti
    start_centroid = tet_centroids[start_ti]
    max_proj       = target_dist * 1.05   # small slack past the child
    cur_proj       = 0.0                  # forward progress already made
    boundary_event = None

    for _ in range(max_steps):
        if cur_ti == goal_ti:
            break

        best_ti, best_proj, is_overshoot = best_directional_neighbour(
            cur_ti, start_centroid, direction, cur_proj, max_proj, visited)

        if best_ti is None:
            break   # genuine dead end — clean stop, no detour

        if not tet_all_negative(best_ti):
            # the one neighbour lying in the parent->child direction is
            # at/over the SDF=0 surface — this is the boundary-touch
            # trigger, NOT a silent stop. Report it; caller handles the
            # detour. The walk itself ends here (the detour logic will
            # continue the journey via fresh recursive corridor walks).
            boundary_event = cur_ti
            break

        corridor.append(best_ti)
        visited.add(best_ti)
        cur_ti   = best_ti
        cur_proj = best_proj

        if cur_ti == goal_ti:
            break

        if is_overshoot:
            # tier-2 fallback step taken (start/goal closer together than
            # one tet hop) — this is the closest the walk can land on the
            # child without a distance bound, so stop here. Taking another
            # tier-2 step would have no distance limit at all (every future
            # step is "overshoot" by definition once cur_proj > max_proj)
            # and could run away arbitrarily far.
            break

    return corridor, boundary_event

# ══════════════════════════════════════════════════════════════════
# 9.  PARAMETERS
# ══════════════════════════════════════════════════════════════════

LINE_SAMPLES   = 11      # samples along START→GOAL line (10 intervals)
N_RAYS         = 8       # rays scattered per branch per round
RAY_LEN        = math.pow(2,0.5)*math.pi
RAY_STEP       = RAY_LEN / 10
N_RAY_SAMPLES  = max(5, int(np.ceil(RAY_LEN / RAY_STEP))) + 1
t_vals_ray     = np.linspace(0.0, RAY_LEN, N_RAY_SAMPLES)

N_PTS_TARGET   = 10     # stop once this many midpoints have been CHOSEN via
                          # scattering (i.e. len(node_names)) — corridor filler
                          # tets shaded between parent/child do NOT count
N_BRANCHES     = 4       # number of parallel branch tips after round 1

rng = np.random.default_rng(seed=42)

def random_unit_dirs(n):
    raw = rng.standard_normal((n, 3))
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)

# ══════════════════════════════════════════════════════════════════
# 10.  NAMED-POINT BOOKKEEPING
# ══════════════════════════════════════════════════════════════════
# Every point we actually KEEP (shade) gets a name ("P0", "P1", ...).
# We only ever store/plot the rays scattered FROM a kept point, never
# the discarded candidate-midpoint cloud or unused boundary points.

node_names   = []          # ordered list of names, in the order shaded
node_cfg     = {}          # name -> cfg (np.array, radians)
node_tet     = {}          # name -> tet index
node_parent  = {}          # name -> parent name (None for root)
node_rays    = {}          # name -> (N_RAYS,3) ray endpoint cfgs actually
                            #         scattered FROM this node (for plotting)
red_tets     = []           # list[int] tet indices, in shading order (kept)

_name_counter = 0
def next_name():
    global _name_counter
    n = f"P{_name_counter}"
    _name_counter += 1
    return n

def register_point(cfg, parent_name, ray_endpoints=None):
    """
    Shade cfg's tet, give it a name, record bookkeeping. Returns name.

    If a parent is given, also walks a corridor of red tets from the
    parent's tet to this point's tet (see walk_corridor). If that walk
    hits a BOUNDARY TOUCH (the tet's directional neighbour toward this
    point has SDF >= 0), the walk doesn't just stop — it spawns a
    detour:
        1. N_PTS_TARGET is bumped up by one (a new midpoint is being
           added to the budget).
        2. The tet where the walk stopped ("current tet") is itself
           registered as a new named point, parented to the ORIGINAL
           parent — this draws the parent->current-tet leg of the
           corridor.
        3. From that new point, 8 rays are scattered exactly like a
           normal growth round; the farthest negative-SDF pairwise
           midpoint is kept and registered as ANOTHER new named point,
           parented to the current-tet point.
        4. The journey then continues: a fresh corridor walk runs from
           that new midpoint toward the ORIGINAL child (cfg/ti) — which
           may itself hit another boundary touch and recurse again.
    So the effective edge becomes:
        parent -> current_tet -> new_midpoint -> ... -> child
    exactly as requested, entirely via recursive corridor walks/registrations.
    """
    global N_PTS_TARGET
    name = next_name()
    ti   = find_containing_tet(cfg)
    node_names.append(name)
    node_cfg[name]    = cfg
    node_tet[name]    = ti
    node_parent[name] = parent_name
    if ray_endpoints is not None:
        node_rays[name] = ray_endpoints
    if ti not in red_tets:
        red_tets.append(ti)

    # ── parent→child tet-walk corridor shading (with detour support) ──
    if parent_name is not None:
        parent_ti  = node_tet[parent_name]
        parent_cfg = node_cfg[parent_name]
        _detour_parent_name[0] = parent_name
        _walk_and_detour(parent_ti, parent_cfg, ti, cfg)

    return name

def _walk_and_detour(from_ti, from_cfg, goal_ti, goal_cfg, _depth=0):
    """
    Runs walk_corridor(from_ti -> goal_ti). On a clean finish, just
    shades the corridor tets. On a boundary touch, spawns the detour
    (current tet -> new scattered midpoint -> named points) and then
    recurses to keep walking from the new midpoint toward goal_ti/
    goal_cfg, so the journey still ultimately reaches the original
    child. _depth is a safety cap against pathological infinite
    detour chains.
    """
    global N_PTS_TARGET
    if _depth > 50:
        print(f"      [WARN] detour recursion cap hit — abandoning "
              f"remainder of this corridor")
        return

    diff = goal_cfg - from_cfg
    dist = np.linalg.norm(diff)
    if dist <= 1e-12 or from_ti == goal_ti:
        return
    direction = diff / dist

    corridor, boundary_event = walk_corridor(from_ti, goal_ti, direction, target_dist=dist)
    for cti in corridor:
        if cti not in red_tets:
            red_tets.append(cti)
    if corridor:
        print(f"      corridor: {len(corridor)} tet(s) shaded "
              f"(depth={_depth})")

    if boundary_event is None:
        return   # clean finish — reached goal / target_dist / dead end

    # ── BOUNDARY TOUCH → spawn detour ──
    touch_ti  = boundary_event
    touch_cfg = tet_centroids[touch_ti]

    N_PTS_TARGET += 1
    print(f"      [BOUNDARY] touch at tet {touch_ti} — "
          f"N_PTS_TARGET -> {N_PTS_TARGET}")

    # 1) register the touching tet itself as a named point, parented to
    #    whichever named point originally started this leg's walk.
    #    We look up the nearest enclosing named parent by walking back
    #    through from_cfg's owner — but from_ti may itself be mid-detour
    #    (not a registered name), so we register relative to whatever
    #    named point this leg conceptually started from. To keep this
    #    simple and correct we pass the *name* down via closures below.
    touch_name = _register_detour_point(touch_cfg, _detour_parent_name[0])

    # 2) scatter rays from the touching point, find farthest -SDF midpoint
    boundary_pts, ray_endpoints = scatter_first_crossings(touch_cfg, N_RAYS)
    candidates = negative_sdf_midpoints(boundary_pts)
    if not candidates:
        print(f"      [BOUNDARY] no valid scatter candidates from tet "
              f"{touch_ti} — detour stalls, corridor ends here")
        return

    best_cfg, best_sdf = max(
        candidates, key=lambda mc: np.linalg.norm(mc[0] - touch_cfg))

    mid_name = _register_detour_point(best_cfg, touch_name, ray_endpoints=ray_endpoints)
    print(f"      [BOUNDARY] new midpoint {mid_name} SDF={best_sdf:+.4f} "
          f"from touch tet {touch_ti}")

    # 3) continue the journey: new midpoint -> original goal, recursively
    mid_ti = node_tet[mid_name]
    _detour_parent_name[0] = mid_name
    _walk_and_detour(mid_ti, best_cfg, goal_ti, goal_cfg, _depth=_depth + 1)


_detour_parent_name = [None]   # 1-elem mutable cell: tracks the "current"
                                 # named parent for chained detour registration

def _register_detour_point(cfg, parent_name, ray_endpoints=None):
    """
    Like register_point, but used internally by _walk_and_detour for
    the two new points a boundary touch introduces (the touch-tet point
    and the new scattered midpoint). Does NOT itself trigger another
    corridor walk from parent_name — _walk_and_detour already manages
    the walk explicitly so we don't double-shade. Still updates
    _detour_parent_name so subsequent detours in the same chain parent
    correctly.
    """
    name = next_name()
    ti   = find_containing_tet(cfg)
    node_names.append(name)
    node_cfg[name]    = cfg
    node_tet[name]    = ti
    node_parent[name] = parent_name
    if ray_endpoints is not None:
        node_rays[name] = ray_endpoints
    if ti not in red_tets:
        red_tets.append(ti)
    _detour_parent_name[0] = name
    return name

# ══════════════════════════════════════════════════════════════════
# 11.  STEP A — LINE SCAN START→GOAL → SEED MIDPOINT (P0 / ROOT)
# ══════════════════════════════════════════════════════════════════

print("\n[INFO] Scanning START→GOAL line for SDF sign changes …")
start_time = time.perf_counter()

line_t    = np.linspace(0.0, 1.0, LINE_SAMPLES)
line_cfgs = [START + t*(GOAL - START) for t in line_t]
line_sdfs = [eval_sdf_vec(c) for c in line_cfgs]

line_boundary_pts = []
for i in range(LINE_SAMPLES - 1):
    if line_sdfs[i] * line_sdfs[i+1] < 0:
        cfg_z, _ = regula_falsi_3d(line_cfgs[i], line_cfgs[i+1],
                                    line_sdfs[i], line_sdfs[i+1])
        line_boundary_pts.append(cfg_z)

print(f"[INFO] Line boundary points found: {len(line_boundary_pts)}")

root_cfg = None
for i in range(len(line_boundary_pts) - 1):
    mid_cfg = (line_boundary_pts[i] + line_boundary_pts[i+1]) / 2.0
    mid_sdf = eval_sdf_vec(mid_cfg)
    if mid_sdf < 0:
        root_cfg = mid_cfg   # last valid seed midpoint becomes ROOT
        print(f"  [line] seed midpoint SDF={mid_sdf:+.4f}")

if root_cfg is None:
    raise RuntimeError(
        "[FATAL] No negative-SDF midpoint found along the START→GOAL line — "
        "cannot seed frontier growth. Check START/GOAL or obstacle placement."
    )

root_name = register_point(root_cfg, parent_name=None)
print(f"[INFO] ROOT seed {root_name} = ({np.degrees(root_cfg[0]):+.1f}°, "
      f"{np.degrees(root_cfg[1]):+.1f}°, {np.degrees(root_cfg[2]):+.1f}°)")

# ══════════════════════════════════════════════════════════════════
# 12.  GROWTH HELPERS
# ══════════════════════════════════════════════════════════════════

def scatter_first_crossings(source_cfg, n_rays):
    """
    Scatter n_rays random-direction rays from source_cfg. Walk each ray
    in N_RAY_SAMPLES steps out to RAY_LEN; stop at the FIRST sign change
    (regula falsi) → at most one boundary point per ray. Rays with no
    sign change contribute nothing.
    Returns: (boundary_pts, ray_endpoints)
      boundary_pts  : list of cfgs where a sign change was found (<=n_rays)
      ray_endpoints : (n_rays,3) array, the FULL endpoint of every ray cast
                      (used only for drawing the 8 rays from this node;
                      independent of whether that ray found a crossing)
    """
    dirs = random_unit_dirs(n_rays)
    ray_endpoints = source_cfg + RAY_LEN * dirs   # full-length ray endpoints
    found = []
    for rdir in dirs:
        cfgs = [source_cfg + t*rdir for t in t_vals_ray]
        sdfs = [eval_sdf_vec(c) for c in cfgs]
        for i in range(len(cfgs) - 1):
            if sdfs[i] * sdfs[i+1] < 0:
                cfg_z, _ = regula_falsi_3d(cfgs[i], cfgs[i+1], sdfs[i], sdfs[i+1])
                found.append(cfg_z)
                break  # first crossing only, then move to next ray
    return found, ray_endpoints

def negative_sdf_midpoints(boundary_pts):
    """
    All pairwise combinations of boundary_pts → midpoint → keep if
    SDF(midpoint) < 0 strictly. Returns list of (midpoint_cfg, sdf).
    Purely a candidate pool — none of this is stored or plotted unless
    a candidate is later chosen by the caller.
    """
    survivors = []
    for cfg_a, cfg_b in itertools.combinations(boundary_pts, 2):
        mid = (cfg_a + cfg_b) / 2.0
        s   = eval_sdf_vec(mid)
        if s < 0:
            survivors.append((mid, s))
    return survivors

# ══════════════════════════════════════════════════════════════════
# 13.  ROUND 1 — ROOT → 4 NAMED BRANCH TIPS (P1..P4)
# ══════════════════════════════════════════════════════════════════

print(f"\n[INFO] Round 1: scattering {N_RAYS} rays from {root_name} …")
round1_boundary, round1_ray_endpoints = scatter_first_crossings(root_cfg, N_RAYS)
node_rays[root_name] = round1_ray_endpoints   # rays drawn FROM the root
print(f"  boundary points found: {len(round1_boundary)}")

round1_candidates = negative_sdf_midpoints(round1_boundary)
print(f"  negative-SDF candidate midpoints: {len(round1_candidates)}")

# rank by distance from ROOT, keep farthest N_BRANCHES
round1_candidates.sort(key=lambda mc: -np.linalg.norm(mc[0] - root_cfg))
round1_keep = round1_candidates[:N_BRANCHES]

branch_tips = []   # list of names, the 4 active branch tips
for mid_cfg, mid_sdf in round1_keep:
    name = register_point(mid_cfg, parent_name=root_name)
    branch_tips.append(name)
    print(f"    kept {name}  SDF={mid_sdf:+.4f}  "
          f"dist_from_root={np.linalg.norm(mid_cfg-root_cfg):.3f}")

print(f"[INFO] After round 1: {len(node_names)} named midpoints chosen "
      f"(target {N_PTS_TARGET}), {len(red_tets)} total red tets, "
      f"{len(branch_tips)} active branches: {branch_tips}")

# ══════════════════════════════════════════════════════════════════
# 14.  ROUND 2+ — 4 PARALLEL BRANCHES UNTIL N_PTS_TARGET
# ══════════════════════════════════════════════════════════════════

round_num = 1
while len(node_names) < N_PTS_TARGET and branch_tips:
    round_num += 1
    next_tips = []

    for tip_name in branch_tips:
        if len(node_names) >= N_PTS_TARGET:
            next_tips.append(tip_name)   # keep tip alive, just stop growing
            continue

        tip_cfg = node_cfg[tip_name]
        boundary_pts, ray_endpoints = scatter_first_crossings(tip_cfg, N_RAYS)

        candidates = negative_sdf_midpoints(boundary_pts)
        if not candidates:
            # this branch found nothing this round — it stays put,
            # we'll try again from the same tip next round
            print(f"  [round {round_num:03d}] {tip_name}: no valid "
                  f"candidates — branch stalled this round")
            next_tips.append(tip_name)
            continue

        # pick candidate farthest from THIS branch's own tip
        best_cfg, best_sdf = max(
            candidates, key=lambda mc: np.linalg.norm(mc[0] - tip_cfg))

        new_name = register_point(best_cfg, parent_name=tip_name,
                                   ray_endpoints=ray_endpoints)
        dist = np.linalg.norm(best_cfg - tip_cfg)
        print(f"  [round {round_num:03d}] {tip_name} → {new_name}  "
              f"SDF={best_sdf:+.4f}  dist_from_parent={dist:.3f}  "
              f"(named: {len(node_names)}/{N_PTS_TARGET}, total red: {len(red_tets)})")
        next_tips.append(new_name)

    branch_tips = next_tips

end_time = time.perf_counter()

print(f"\n[INFO] Growth done in {end_time - start_time:.2f}s")
print(f"  Named midpoints  : {len(node_names)} (target {N_PTS_TARGET})")
print(f"  Total red tets   : {len(red_tets)} (named + corridor filler)")
print(f"  Rounds run       : {round_num}")
print(f"  Named points     : {len(node_names)}")
print(f"  SDF queries      : {_sdf_query_count}  (cache misses only)")

# ══════════════════════════════════════════════════════════════════
# 14b.  BUILD RED TET MESH
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
# 15.  BUILD "KEPT RAYS ONLY" GEOMETRY
# ══════════════════════════════════════════════════════════════════
# Only rays that were actually scattered FROM a kept/named point are
# drawn (node_rays). No discarded candidate-midpoint cloud is plotted.
# Parent→child connections are NEVER drawn as explicit lines — the
# tet-walk corridor (a chain of shaded red tets, see walk_corridor)
# IS the visual representation of that connection, already baked into
# red_tets / red_mesh above like any other shaded tet.

def build_ray_mesh():
    """8 rays per node that has stored ray_endpoints, in degree-space."""
    all_pts, line_cells, pt_offset = [], [], 0
    for name, endpoints in node_rays.items():
        src_deg = np.degrees(node_cfg[name])
        ends_deg = np.degrees(endpoints)
        for end in ends_deg:
            seg = np.array([src_deg, end])
            all_pts.append(seg)
            line_cells.append(np.hstack([[2], [pt_offset, pt_offset+1]]))
            pt_offset += 2
    if not all_pts:
        return None
    all_pts = np.vstack(all_pts)
    lines   = np.concatenate(line_cells).astype(np.int64)
    mesh = pv.PolyData(all_pts)
    mesh.lines = lines
    return mesh

ray_mesh  = build_ray_mesh()

# ══════════════════════════════════════════════════════════════════
# 16.  VISUALISE
# ══════════════════════════════════════════════════════════════════

pl = pv.Plotter(window_size=[1400, 900])
pl.set_background("white")

if red_mesh is not None:
    pl.add_mesh(red_mesh, color="#C0392B", opacity=0.85,
                show_edges=True, edge_color="#7B241C", line_width=0.4,
                label=f"Shaded tets ({len(red_tets)})")

if ray_mesh is not None:
    pl.add_mesh(ray_mesh, color="#95A5A6", line_width=1.0, opacity=0.4,
                label=f"Scattered rays ({N_RAYS}/node)")

# START→GOAL reference line
pl.add_mesh(pv.Spline(np.degrees(np.array(
    [START + t*(GOAL-START) for t in np.linspace(0,1,200)]
)), 200), color="dodgerblue", line_width=4, label="START→GOAL line")

pl.add_points(np.degrees(START).reshape(1,3), color="lime",   point_size=22,
              render_points_as_spheres=True, label="Start")
pl.add_points(np.degrees(GOAL).reshape(1,3),  color="orange", point_size=22,
              render_points_as_spheres=True, label="Goal")

# All KEPT/named points, colour-coded: root magenta, others by branch
named_pts_deg = np.degrees(np.array([node_cfg[n] for n in node_names]))
pl.add_points(named_pts_deg, color="#8E44AD", point_size=14,
              render_points_as_spheres=True, label=f"Named points ({len(node_names)})")
pl.add_points(np.degrees(root_cfg).reshape(1,3), color="magenta", point_size=22,
              render_points_as_spheres=True, label=f"Root ({root_name})")

# Text labels for each named point (small, only if not too many to read)
if len(node_names) <= 60:
    for name in node_names:
        pl.add_point_labels(np.degrees(node_cfg[name]).reshape(1,3), [name],
                             font_size=10, text_color="black",
                             shape=None, always_visible=True)

pl.add_axes(xlabel="q1 (deg)", ylabel="q2 (deg)", zlabel="q3 (deg)")
pl.add_legend(bcolor="white", border=True, size=(0.30, 0.20))
pl.add_title(
    f"3D C-Space | Branch-tree growth | named={len(node_names)}/{N_PTS_TARGET} | "
    f"red_tets={len(red_tets)} | "
    f"rounds={round_num} | branches={N_BRANCHES} | rays/node={N_RAYS} | "
    f"queries={_sdf_query_count} | time={end_time-start_time:.2f}s",
    font_size=10)

print("[INFO] Opening PyVista window …")
pl.show()
p.disconnect()
print("[INFO] Done.")