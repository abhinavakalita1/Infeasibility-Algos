"""
surface_sdf_pipeline.py

Pipeline:
  1. Load ground-truth obstacle boxes (analytical primitives) into PyBullet
  2. Load pre-built reconstructed meshes from objs/*.obj as collision bodies
  3. Sample 10 000 random C-space configs
     → query p.getClosestPoints on reconstructed mesh  (our method)
     → query p.getClosestPoints on original obstacle   (ground truth)
  4. Display accuracy results + 8 plots
"""

import pybullet_data
import numpy as np
import pybullet as p
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
import json
import glob


# ══════════════════════════════════════════════════════════════
# 1.  PYBULLET SETUP
# ══════════════════════════════════════════════════════════════

physicsClient = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
planeId = p.loadURDF("plane.urdf")

armId = p.loadURDF(
    "arm_5.urdf",
    [0, 0, 0],
    p.getQuaternionFromEuler([0, 0, 0]),
    useFixedBase=True,
    flags=(
        p.URDF_USE_INERTIA_FROM_FILE |
        p.URDF_USE_SELF_COLLISION     |
        p.URDF_USE_IMPLICIT_CYLINDER
    )
)
NUM_JOINTS = p.getNumJoints(armId)
print(f"[INFO] Arm loaded — {NUM_JOINTS} joints")


# ══════════════════════════════════════════════════════════════
# 2.  LOAD ORIGINAL OBSTACLE  (ground truth body)
# ══════════════════════════════════════════════════════════════

def load_mesh_obstacle(obj_path, position=[0,0,0], orientation=[0,0,0],
                       scale=1.0, color=[0.8,0.5,0.2,1]):
    col = p.createCollisionShape(
        p.GEOM_MESH, fileName=obj_path,
        meshScale=[scale, scale, scale],
        flags=p.GEOM_FORCE_CONCAVE_TRIMESH
    )
    vis = p.createVisualShape(
        p.GEOM_MESH, fileName=obj_path,
        meshScale=[scale, scale, scale],
        rgbaColor=color
    )
    return p.createMultiBody(0, col, vis, position,
                             p.getQuaternionFromEuler(orientation))

# concaveId    = load_mesh_obstacle("concave.obj", position=[0, 0.5, 0], scale=0.4)
# obstacle_ids = [concaveId]
# print(f"[INFO] Ground truth obstacle loaded (body_id={concaveId})")

def create_box(half_extents=[1,1,1], position=[0,0,0], orientation=[0,0,0], mass=0, color=[1,0,0,1]):
    col  = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    vis  = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
    quat = p.getQuaternionFromEuler(orientation)
    return p.createMultiBody(mass, col, vis, position, quat)


box2Id      = create_box(half_extents=[1,.2,1],      position=[0,.8,.8],      orientation=[0, 0, 0])
box3Id      = create_box(half_extents=[1,.2,1],      position=[0,-.8,.8],      orientation=[0, 0, 0])
box4Id      = create_box(half_extents=[.2,1,1],      position=[.8,0,.8],      orientation=[0, 0, 0])
box5Id      = create_box(half_extents=[.2,1,1],      position=[-.8,0,.8],      orientation=[0, 0, 0])

obstacle_ids   = [box2Id, box3Id, box4Id, box5Id]


# ══════════════════════════════════════════════════════════════
# 3.  LOAD RECONSTRUCTED OBSTACLE MESHES (objs/)
# ══════════════════════════════════════════════════════════════

#Getting objs
recon_body_ids  = []
recon_obj_paths = []

print("[INFO] Loading reconstructed obstacle meshes from objs/ ...")
t_recon_start = time.perf_counter()

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
print(f"[INFO] {len(recon_body_ids)} mesh bodies loaded into PyBullet")

if len(recon_body_ids) == 0:
    raise RuntimeError(
        "No mesh bodies were loaded — check that objs/ contains .obj files."
    )


# ══════════════════════════════════════════════════════════════
# 6.  HELPERS
# ══════════════════════════════════════════════════════════════

# Use a tight threshold — broad thresholds force PyBullet to sweep
# the entire broadphase and return huge contact lists.
# Tune QUERY_DIST to the largest clearance you care about.
QUERY_DIST = 1.5   # metres — adjust to your scene scale


def set_config(config):
    for j, angle in enumerate(config):
        p.resetJointState(armId, j, float(angle))


def sdf_recon(arm_id):
    """Closest signed distance against reconstructed alpha-mesh bodies."""
    min_d = QUERY_DIST
    for body_id in recon_body_ids:
        contacts = p.getClosestPoints(
            bodyA=arm_id, bodyB=body_id, distance=QUERY_DIST
        )
        if contacts:
            d = min(c[8] for c in contacts)
            if d < min_d:
                min_d = d
            if min_d < 0:          # already in collision — can't get worse
                return min_d
    return min_d


def sdf_gt(arm_id):
    """Closest signed distance against original obstacle (ground truth)."""
    min_d = QUERY_DIST
    for obs_id in obstacle_ids:
        contacts = p.getClosestPoints(
            bodyA=arm_id, bodyB=obs_id, distance=QUERY_DIST
        )
        if contacts:
            d = min(c[8] for c in contacts)
            if d < min_d:
                min_d = d
            if min_d < 0:
                return min_d
    return min_d


# ══════════════════════════════════════════════════════════════
# 7.  SAMPLE 10 000 RANDOM C-SPACE CONFIGS + COMPARE
# ══════════════════════════════════════════════════════════════

N_TOTAL  = 10000
INTERVAL = 1000

print(f"\n[INFO] Sampling {N_TOTAL} random configs…\n")

configs    = np.random.uniform(-np.pi, np.pi, size=(N_TOTAL, NUM_JOINTS))
recon_dist = np.zeros(N_TOTAL)
gt_dist    = np.zeros(N_TOTAL)

t_recon_q = 0.0
t_gt_q    = 0.0

interval_mae       = []
interval_rmse      = []
interval_max       = []
interval_bias      = []
interval_col_agree = []
interval_idx       = []

for i, cfg_joints in enumerate(configs):
    set_config(cfg_joints)
    p.stepSimulation()

    t0 = time.perf_counter()
    recon_dist[i] = sdf_recon(armId)
    t_recon_q += time.perf_counter() - t0

    t0 = time.perf_counter()
    gt_dist[i] = sdf_gt(armId)
    t_gt_q += time.perf_counter() - t0

    if (i + 1) % INTERVAL == 0:
        sl    = slice(i + 1 - INTERVAL, i + 1)
        err   = np.abs(recon_dist[sl] - gt_dist[sl])
        bias  = recon_dist[sl] - gt_dist[sl]
        agree = float(np.mean((recon_dist[sl] < 0) == (gt_dist[sl] < 0))) * 100

        interval_mae.append(float(np.mean(err)))
        interval_rmse.append(float(np.sqrt(np.mean(err**2))))
        interval_max.append(float(np.max(err)))
        interval_bias.append(float(np.mean(bias)))
        interval_col_agree.append(agree)
        interval_idx.append(i + 1)

        print(f"  [{i+1:5d}]  "
              f"MAE={interval_mae[-1]:.4f}m  "
              f"RMSE={interval_rmse[-1]:.4f}m  "
              f"MaxErr={interval_max[-1]:.4f}m  "
              f"Bias={interval_bias[-1]:+.4f}m  "
              f"Agree={agree:.1f}%")

all_err  = np.abs(recon_dist - gt_dist)
all_bias = recon_dist - gt_dist
rec_col  = recon_dist < 0
gt_col   = gt_dist    < 0
speedup  = t_gt_q / max(t_recon_q, 1e-9)

print(f"\n{'═'*60}")
print(f"  GLOBAL ACCURACY  (N={N_TOTAL})")
print(f"  Method: Pre-built mesh reconstruction (objs/)")
print(f"{'═'*60}")
print(f"  MAE                         : {np.mean(all_err):.5f} m")
print(f"  RMSE                        : {np.sqrt(np.mean(all_err**2)):.5f} m")
print(f"  Max absolute error          : {np.max(all_err):.5f} m")
print(f"  Mean bias (Recon - GT)      : {np.mean(all_bias):+.5f} m")
print(f"  Collision agreement         : {np.mean(rec_col == gt_col)*100:.2f}%")
print(f"  False positives             : {np.sum(rec_col & ~gt_col)}")
print(f"  False negatives             : {np.sum(~rec_col & gt_col)}")
print(f"  Recon mesh time / query     : {t_recon_q/N_TOTAL*1000:.3f} ms")
print(f"  GT time / query             : {t_gt_q/N_TOTAL*1000:.3f} ms")
print(f"  Speedup (recon vs GT)       : {speedup:.2f}x")
print(f"{'═'*60}")
print(f"\n  Pipeline timing:")
print(f"    Mesh loading (objs/)      : {t_recon:.1f} ms  (one-time)")


# ══════════════════════════════════════════════════════════════
# 8.  PLOTS
# ══════════════════════════════════════════════════════════════

BLUE   = "#3A7DC9"
ORANGE = "#E8882A"
RED    = "#C93A3A"
GREEN  = "#2E9E5B"
PURPLE = "#7F3FBF"
LBLUE  = "#B5D4F4"
GRAY   = "#888780"

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
})

xs = interval_idx


# ── Plot 1: MAE & RMSE ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(xs, interval_mae,  "o-",  color=BLUE,   lw=2, ms=7, label="MAE")
ax.plot(xs, interval_rmse, "s--", color=ORANGE, lw=2, ms=7, label="RMSE")
ax.set_title(f"MAE and RMSE per 1 000-config interval\n"
             f"(Pre-built mesh reconstruction)")
ax.set_xlabel("Configs evaluated")
ax.set_ylabel("Error  (m)")
ax.set_xticks(xs)
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig("plot1_mae_rmse.png", dpi=150, bbox_inches="tight")
plt.show()


# ── Plot 2: Max error ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(xs, interval_max, width=700, color=RED, edgecolor="white", lw=0.5)
for x, v in zip(xs, interval_max):
    ax.text(x, v + 0.01, f"{v:.3f}", ha="center", va="bottom",
            fontsize=9, color=RED)
ax.set_title("Max absolute error per 1 000-config interval")
ax.set_xlabel("Configs evaluated")
ax.set_ylabel("|Recon − GT|  max  (m)")
ax.set_xticks(xs)
plt.tight_layout()
plt.savefig("plot2_max_error.png", dpi=150, bbox_inches="tight")
plt.show()


# ── Plot 3: Bias ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
colors = [GREEN if b >= 0 else RED for b in interval_bias]
ax.bar(xs, interval_bias, width=700, color=colors, edgecolor="white", lw=0.5)
ax.axhline(0, color="black", lw=0.8, zorder=3)
for x, v in zip(xs, interval_bias):
    va = "bottom" if v >= 0 else "top"
    ax.text(x, v + (0.001 if v >= 0 else -0.001),
            f"{v:+.4f}", ha="center", va=va, fontsize=9)
ax.set_title("Mean signed bias (Recon − GT) per interval")
ax.set_xlabel("Configs evaluated")
ax.set_ylabel("Bias  (m)   [+ = Recon overestimates, safe]")
ax.set_xticks(xs)
ax.legend(handles=[Patch(color=GREEN, label="Recon overestimates (safe)"),
                   Patch(color=RED,   label="Recon underestimates (unsafe)")],
          frameon=False, fontsize=9)
plt.tight_layout()
plt.savefig("plot3_bias.png", dpi=150, bbox_inches="tight")
plt.show()


# ── Plot 4: Collision agreement ──────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(xs, interval_col_agree, "D-", color=PURPLE, lw=2, ms=8, zorder=3)
ax.fill_between(xs, interval_col_agree, alpha=0.12, color=PURPLE)
ax.axhline(100, color=GREEN, lw=1.2, ls="--", label="Perfect (100%)")
ax.set_ylim(0, 105)
ax.set_title("Collision detection agreement per interval")
ax.set_xlabel("Configs evaluated")
ax.set_ylabel("Agreement  (%)")
ax.set_xticks(xs)
for x, v in zip(xs, interval_col_agree):
    ax.text(x, v - 3.5, f"{v:.1f}%", ha="center", va="top",
            fontsize=9, color=PURPLE)
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig("plot4_collision_agree.png", dpi=150, bbox_inches="tight")
plt.show()


# ── Plot 5: Recon vs GT scatter ──────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 6.5))
lim = max(np.abs(recon_dist).max(), np.abs(gt_dist).max()) * 1.05
ax.scatter(gt_dist, recon_dist, s=1.5, alpha=0.12,
           color=BLUE, rasterized=True)
ax.plot([-lim, lim], [-lim, lim], color=RED, lw=1.5, ls="--",
        label="Perfect  y = x")
ax.axhline(0, color="black", lw=0.5, alpha=0.4)
ax.axvline(0, color="black", lw=0.5, alpha=0.4)
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_aspect("equal")
ax.set_title(f"Alpha recon SDF vs GT — all configs")
ax.set_xlabel("GT  p.getClosestPoints (original mesh)  (m)")
ax.set_ylabel("Recon  p.getClosestPoints (mesh)  (m)")
ax.legend(frameon=False)
ax.text( lim*0.55,  lim*0.82, "Both free",      fontsize=9, color=GRAY,   ha="center")
ax.text(-lim*0.55, -lim*0.82, "Both collision",  fontsize=9, color=GRAY,   ha="center")
ax.text(-lim*0.55,  lim*0.82, "False positive",  fontsize=9, color=RED,    ha="center")
ax.text( lim*0.55, -lim*0.82, "False negative",  fontsize=9, color=ORANGE, ha="center")
plt.tight_layout()
plt.savefig("plot5_scatter.png", dpi=150, bbox_inches="tight")
plt.show()


# ── Plot 6: Error histogram ──────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(all_err, bins=80, color=LBLUE, edgecolor=BLUE, linewidth=0.3)
ax.axvline(np.mean(all_err),   color=RED,    lw=2, ls="--",
           label=f"MAE    = {np.mean(all_err):.4f} m")
ax.axvline(np.median(all_err), color=ORANGE, lw=2, ls=":",
           label=f"Median = {np.median(all_err):.4f} m")
ax.set_title("Absolute error distribution — all configs")
ax.set_xlabel("|Recon SDF − GT|  (m)")
ax.set_ylabel("Count")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig("plot6_error_hist.png", dpi=150, bbox_inches="tight")
plt.show()


# ── Plot 7: Timing ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 4.5))
methods = ["Recon mesh\n(objs/)", "p.getClosestPoints\n(original GT)"]
times   = [t_recon_q/N_TOTAL*1000, t_gt_q/N_TOTAL*1000]
bars    = ax.bar(methods, times, color=[BLUE, RED],
                 edgecolor="white", lw=0.5, width=0.45)
for bar, t in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width()/2,
            t + 0.005, f"{t:.3f} ms",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_title(f"Mean query time per config  (N = {N_TOTAL:,})")
ax.set_ylabel("Time  (ms / config)")
faster = "Recon" if t_recon_q < t_gt_q else "GT"
ratio  = max(speedup, 1/speedup if speedup > 0 else 1)
ax.text(0.98, 0.96, f"{faster} is {ratio:.1f}× faster",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=10, color=BLUE,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=BLUE, lw=0.8))
plt.tight_layout()
plt.savefig("plot7_timing.png", dpi=150, bbox_inches="tight")
plt.show()


# ── Plot 8: Confusion matrix ─────────────────────────────────
fp = int(np.sum(rec_col & ~gt_col))
fn = int(np.sum(~rec_col & gt_col))
tp = int(np.sum(rec_col & gt_col))
tn = int(np.sum(~rec_col & ~gt_col))

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

cm        = np.array([[tn, fp], [fn, tp]])
labels_cm = [["True Neg\n(both free)",     "False Pos\n(Recon coll,\nGT free)"],
             ["False Neg\n(Recon free,\nGT coll)", "True Pos\n(both coll)"]]
colors_cm = [[GREEN, RED], [ORANGE, GREEN]]

ax = axes[0]
ax.set_xlim(0, 2); ax.set_ylim(0, 2)
ax.set_xticks([0.5, 1.5]); ax.set_xticklabels(["GT: Free", "GT: Collision"])
ax.set_yticks([0.5, 1.5]); ax.set_yticklabels(["Recon: Collision", "Recon: Free"])
ax.set_title("Confusion matrix  (Recon mesh vs GT)")
ax.spines[:].set_visible(False)
ax.grid(False)
for r in range(2):
    for c in range(2):
        val = cm[r, c]
        ax.add_patch(plt.Rectangle((c, 1-r), 1, 1,
                                   fc=colors_cm[r][c], alpha=0.25,
                                   ec="white", lw=2))
        ax.text(c + 0.5, 1 - r + 0.5,
                f"{labels_cm[r][c]}\n{val:,}\n({val/N_TOTAL*100:.1f}%)",
                ha="center", va="center", fontsize=9)

ax2    = axes[1]
cats   = ["True\nNeg", "True\nPos", "False\nPos", "False\nNeg"]
counts = [tn, tp, fp, fn]
cols   = [GREEN, GREEN, RED, ORANGE]
bars2  = ax2.bar(cats, counts, color=cols, edgecolor="white", lw=0.5, width=0.5)
for bar, v in zip(bars2, counts):
    ax2.text(bar.get_x() + bar.get_width()/2,
             v + N_TOTAL*0.005, f"{v:,}",
             ha="center", va="bottom", fontsize=10, fontweight="bold")
ax2.set_title("Classification counts")
ax2.set_ylabel("Count")

plt.tight_layout()
plt.savefig("plot8_confusion.png", dpi=150, bbox_inches="tight")
plt.show()


p.disconnect()
print("\n[INFO] All done.")
print(f"  Mesh files saved: {recon_obj_paths}")
print(f"  → plot1_mae_rmse.png  → plot2_max_error.png")
print(f"  → plot3_bias.png      → plot4_collision_agree.png")
print(f"  → plot5_scatter.png   → plot6_error_hist.png")
print(f"  → plot7_timing.png    → plot8_confusion.png")