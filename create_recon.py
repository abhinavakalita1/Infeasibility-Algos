import pybullet_data
import numpy as np
from sklearn.cluster import DBSCAN
import open3d as o3d
import pybullet as p
import shutil, os, json, glob
import time

t = time.time()

# Pybullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
p.loadURDF("plane.urdf")

arm3Id = p.loadURDF(
    "arm_3.urdf", basePosition=[0,0,0],
    baseOrientation=p.getQuaternionFromEuler([0,0,0]),
    useFixedBase=True,
    flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION)
print(f"[INFO] Arm loaded — {p.getNumJoints(arm3Id)} joints")


# Load points
try:
    points = np.load("points.npy")
    print(f"[INFO] Loaded points.npy ({len(points)} pts)")
except FileNotFoundError:
    raise FileNotFoundError("points.npy not found.")


#Clustering
CONFIG_FILE = "dbscan_config.json"
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE) as f: params = json.load(f)
    eps, min_samples = params["eps"], params["min_samples"]
else:
    raise FileNotFoundError("config.json not found.")

def group(points, labels, min_points_threshold=100):
    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_sizes  = dict(zip(unique_labels, counts))
    small_clusters = [lbl for lbl,cnt in cluster_sizes.items() if lbl != -1 and cnt < min_points_threshold]
    for small_lbl in small_clusters:
        small_mask = labels == small_lbl
        small_pts  = points[small_mask]
        distances  = []
        for lbl in unique_labels:
            if lbl == small_lbl or lbl == -1: continue
            mask = labels == lbl
            if not mask.any(): continue
            centroid = points[mask].mean(axis=0)
            distances.append((lbl, np.mean(np.linalg.norm(small_pts - centroid, axis=1))))
        if distances:
            closest = min(distances, key=lambda x: x[1])[0]
            labels[small_mask] = closest
    return labels

db     = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
labels = db.labels_.copy()
labels = group(points, labels)
unique_labels = np.unique(labels[labels >= 0])
cluster_pts   = [points[labels == lbl] for lbl in unique_labels]

#Helper
def simplify_mesh(mesh, target_triangles=200):
    if len(mesh.triangles) <= target_triangles:
        return mesh
    return mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)


#Alpha reconstruction
OBJ_DIR = "objs"
if os.path.exists(OBJ_DIR):
    shutil.rmtree(OBJ_DIR)
os.makedirs(OBJ_DIR)

ALPHA = 0.4
print(f"[INFO] Alpha surface reconstruction (alpha={ALPHA})…")
for i, cpts in enumerate(cluster_pts):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cpts.astype(np.float64))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=30))
    centre = cpts.mean(axis=0)
    pcd.orient_normals_towards_camera_location(centre + np.array([0, 0, 5.0]))
    try:
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=ALPHA)
    except Exception:
        continue
    if len(mesh.triangles) == 0:
        continue

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    # --- simplification goes here ---
    mesh = simplify_mesh(mesh, target_triangles=150)   # see below

    mesh.compute_vertex_normals()
    obj_path = os.path.join(OBJ_DIR, f"recon_cluster_boundary_{i}.obj")
    o3d.io.write_triangle_mesh(obj_path, mesh)

#Getting objs
recon_body_ids = []

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

        recon_body_ids.append(body_id)

    except Exception as e:
        print(f"Failed to load {obj_path}: {e}")

print(time.time()-t)