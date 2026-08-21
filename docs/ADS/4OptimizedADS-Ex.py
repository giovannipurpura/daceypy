from inspect import getsourcefile
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from numpy.typing import NDArray
from daceypy import ADS, DA, array, RK
from daceypy import ADSintegrator, ADSintegrator_optimized
from daceypy import ADS_utils
from daceypy._ADSintegrator import ADSstate
import time
import itertools
import os
from scipy.spatial import ConvexHull
from matplotlib.collections import LineCollection


def CR3BP(x: array, t: float, mu: float) -> array:
    """
    Circular Restricted Three Body Problem dynamics in normalized coordinates.
    
    Args:
        x: State vector [x, y, z, vx, vy, vz]
        t: Time (unused in autonomous system)
        mu: Mass parameter (Earth-Moon: ~0.01215)
    
    Returns:
        State derivative [vx, vy, vz, ax, ay, az]
    """
    # Extract position and velocity
    pos: array = x[:3]
    vel: array = x[3:]
    
    x_pos, y_pos, z_pos = pos[0], pos[1], pos[2]
    
    # Distances from the two primaries
    r1 = ((x_pos + mu)**2 + y_pos**2 + z_pos**2)**0.5
    r2 = ((x_pos - 1 + mu)**2 + y_pos**2 + z_pos**2)**0.5
    
    # Pseudo-potential gradient
    Omega_x = x_pos - (1 - mu) * (x_pos + mu) / r1**3 - mu * (x_pos - 1 + mu) / r2**3
    Omega_y = y_pos - (1 - mu) * y_pos / r1**3 - mu * y_pos / r2**3
    Omega_z = -(1 - mu) * z_pos / r1**3 - mu * z_pos / r2**3
    
    # Accelerations (Coriolis + centrifugal + gravitational)
    ax = 2 * vel[1] + Omega_x
    ay = -2 * vel[0] + Omega_y
    az = Omega_z
    
    acc = array([ax, ay, az])
    
    # State derivative
    dx = vel.concat(acc)
    return dx


class AutomaticADS_CR3BP_integrator_optimized(ADSintegrator_optimized):
    """
    Custom child class of ADSintegrator_optimized for CR3BP dynamics.
    """
    def __init__(self, mu: float, RK: RK.RKCoeff = RK.RK78()):
        super(AutomaticADS_CR3BP_integrator_optimized, self).__init__(RK)
        self.mu = mu
    
    def f(self, x, t):
        return CR3BP(x, t, self.mu)


class AutomaticADS_CR3BP_integrator(ADSintegrator):
    """
    Custom child class of ADSintegrator for CR3BP dynamics.
    """
    def __init__(self, mu: float, RK: RK.RKCoeff = RK.RK78()):
        super(AutomaticADS_CR3BP_integrator, self).__init__(RK)
        self.mu = mu
    
    def f(self, x, t):
        return CR3BP(x, t, self.mu)



def _local_coordinates(
    pts: np.ndarray,
    x0: np.ndarray,
    rotation: np.ndarray | None = None,
    basis_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """
    Map physical points to local coordinates around x0.

    If basis_matrix is provided, it is interpreted as the columns of the
    affine basis matrix B such that x = x0 + B u and therefore u is obtained
    by solving B u = x - x0.

    If rotation is provided instead, it is interpreted as an orthonormal
    rotation matrix and the coordinates are computed as (x - x0) @ rotation.T.
    """
    pts = np.asarray(pts, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    x0 = np.asarray(x0, dtype=float).ravel()
    centered = pts - x0[np.newaxis, :]

    if basis_matrix is not None:
        B = np.asarray(basis_matrix, dtype=float)
        if B.shape[0] != B.shape[1]:
            raise ValueError("basis_matrix must be square.")
        if B.shape[0] != centered.shape[1]:
            raise ValueError("basis_matrix dimension must match points dimension.")
        return np.linalg.solve(B, centered.T).T

    if rotation is not None:
        R = np.asarray(rotation, dtype=float)
        return centered @ R.T

    return centered


def _get_box_corners(box):
    """
    Extract the corner cloud from an ADS box description.

    Supports the common shapes used in this repo:
    - box["physical"]["corners"]
    - box["corners"]
    - box["box"]["corners"]
    - fallback from box["min"], box["max"]
    """
    if isinstance(box, dict):
        if "physical" in box and isinstance(box["physical"], dict) and "corners" in box["physical"]:
            return np.asarray(box["physical"]["corners"], dtype=float)
        if "corners" in box:
            return np.asarray(box["corners"], dtype=float)
        if "box" in box and isinstance(box["box"], dict) and "corners" in box["box"]:
            return np.asarray(box["box"]["corners"], dtype=float)
        if "min" in box and "max" in box:
            min_v = np.asarray(box["min"], dtype=float)
            max_v = np.asarray(box["max"], dtype=float)
            ndim = min_v.shape[0]
            corners = []
            for bits in itertools.product([0, 1], repeat=ndim):
                mask = np.asarray(bits, dtype=bool)
                corners.append(np.where(mask, max_v, min_v))
            return np.asarray(corners, dtype=float)
    raise KeyError("Could not extract corners from the provided box.")
 

# ─────────────────────────────────────────────────────────────────────────────
# Public function
# ─────────────────────────────────────────────────────────────────────────────
 
def plot_ADS_boxes_2D(
    domain_boxes,
    points=None,
    rotation=None,
    basis_matrix=None,
    x0=None,
    title_prefix="ADS domains",
    ndim=6,
    save=False,
    save_path=".",
    save_format="png",
    dpi=150,
):
    """
    Plot ADS boxes using only their corners.

    The geometry is transformed around x0.
    If basis_matrix is provided, it is treated as a general affine basis B
    with physical coordinates x = x0 + B u and local coordinates obtained by
    solving B u = x - x0.
    Otherwise rotation is treated as an orthonormal rotation:
        y = (x - x0) @ rotation.T

    The figure shows all pairwise combinations:
        (dim i vs dim j) for every i != j

    Points, when provided, are transformed with the same map.
    No box suppression is applied.
    """
    cmap = mpl.colormaps.get_cmap("tab10")

    def _unwrap(entry):
        return entry["geometry"] if isinstance(entry, dict) and "geometry" in entry else entry

    boxes = [_unwrap(d) for d in domain_boxes]

    if points is not None:
        pts = np.asarray(points, dtype=float)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)
    else:
        pts = None

    if pts is not None and x0 is None:
        x0 = np.zeros(pts.shape[1], dtype=float)
    elif x0 is None and boxes:
        first_corners = np.asarray(_get_box_corners(boxes[0]), dtype=float)
        x0 = np.zeros(first_corners.shape[1], dtype=float)
    else:
        x0 = np.asarray(x0, dtype=float).ravel()

    if rotation is not None:
        R = np.asarray(rotation, dtype=float)
    else:
        R = None

    if pts is not None:
        pts_rot = _local_coordinates(pts, x0, rotation=R, basis_matrix=basis_matrix)
    else:
        pts_rot = None
 
    if ndim < 2:
        raise ValueError("ndim must be at least 2.")

    n_panels = ndim
    panel_size = min(4.2, max(2.5, 18.0 / max(n_panels, 1)))
    fig, axes = plt.subplots(
        n_panels,
        n_panels,
        figsize=(panel_size * n_panels, panel_size * n_panels),
        squeeze=False,
    )

    rotated_boxes = []
    for j_domain, d in enumerate(boxes):
        corners = np.asarray(_get_box_corners(d), dtype=float)
        corners_rot = _local_coordinates(corners, x0, rotation=R, basis_matrix=basis_matrix)
        rotated_boxes.append(corners_rot)

    for i in range(n_panels):
        for j in range(n_panels):
            ax = axes[i, j]
            if j >= i:
                ax.axis("off")
                continue

            dim_x = j
            dim_y = i
            if dim_y >= ndim:
                ax.axis("off")
                continue

            scene_points = []
            line_segments = []

            for j_domain, corners_rot in enumerate(rotated_boxes):
                color = cmap(j_domain % 10)
                if corners_rot.shape[1] <= dim_y:
                    continue

                xy = corners_rot[:, [dim_x, dim_y]]
                scene_points.append(xy)

                try:
                    hull = ConvexHull(xy)
                    order = hull.vertices
                except Exception:
                    order = np.arange(xy.shape[0])
                if order.size >= 2:
                    closed = np.vstack([xy[order], xy[order[0]]])
                    line_segments.append(closed)

            if pts_rot is not None:
                ax.scatter(
                    pts_rot[:, dim_x],
                    pts_rot[:, dim_y],
                    color="tab:red",
                    s=10,
                    edgecolors="none",
                    alpha=0.55,
                    rasterized=True,
                )

            if line_segments:
                ax.add_collection(LineCollection(line_segments, colors="black", linewidths=0.9, alpha=0.75))

            ax.set_xlabel(
                f"Dim {dim_x + 1}" if i == n_panels - 1 else "",
                fontweight="bold",
                labelpad=12,
            )
            ax.set_ylabel(
                f"Dim {dim_y + 1}" if j == 0 else "",
                fontweight="bold",
                labelpad=12,
            )
            if i != n_panels - 1:
                ax.tick_params(axis="x", labelbottom=False, bottom=False)
            if j != 0:
                ax.tick_params(axis="y", labelleft=False, left=False)
            ax.set_title("")
            ax.grid(True, alpha=0.25)

            if scene_points:
                all_scene_points = np.vstack(scene_points)
                mins = all_scene_points.min(axis=0)
                maxs = all_scene_points.max(axis=0)
                span_x = maxs[0] - mins[0]
                span_y = maxs[1] - mins[1]
                eps = 1e-9
                if span_x <= 0:
                    center_x = mins[0]
                    span_x = eps
                    mins_x = center_x - 0.5 * span_x
                    maxs_x = center_x + 0.5 * span_x
                else:
                    pad_x = 0.05 * span_x
                    mins_x = mins[0] - pad_x
                    maxs_x = maxs[0] + pad_x

                if span_y <= 0:
                    center_y = mins[1]
                    span_y = eps
                    mins_y = center_y - 0.5 * span_y
                    maxs_y = center_y + 0.5 * span_y
                else:
                    pad_y = 0.05 * span_y
                    mins_y = mins[1] - pad_y
                    maxs_y = maxs[1] + pad_y

                x_center = 0.5 * (mins_x + maxs_x)
                y_center = 0.5 * (mins_y + maxs_y)
                half_span = 0.5 * max(maxs_x - mins_x, maxs_y - mins_y)
                ax.set_xlim(x_center - half_span, x_center + half_span)
                ax.set_ylim(y_center - half_span, y_center + half_span)
            ax.set_aspect("equal", adjustable="box")

    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.08, top=0.97, wspace=0.12, hspace=0.12)
    plt.tight_layout(rect=(0.05, 0.05, 0.99, 0.98))

    if save:
        if os.path.splitext(save_path)[1]:
            base, ext = os.path.splitext(save_path)
            filepath = f"{base}_dim1_vs_all{ext}"
        else:
            os.makedirs(save_path, exist_ok=True)
            filename = f"{title_prefix.replace(' ', '_')}_dim1_vs_all.{save_format}"
            filepath = os.path.join(save_path, filename)
        fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {filepath}")
        plt.close(fig)
    else:
        plt.show()

def plot_domain_evolution(listOut_opt: List[List[ADSstate]], t_eval: np.ndarray):
    """
    Plot number of domains evolution over time for OPTIMIZED integrator.
    """
    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()
    
    # Number of domains over time
    n_domains_opt = [len(states) for states in listOut_opt]
    
    ax.plot(t_eval, n_domains_opt, 'b-o', linewidth=2, markersize=8, label='Optimized - Number of Domains')
    ax.set_xlabel('Time (normalized)', fontsize=12)
    ax.set_ylabel('Number of Domains', fontsize=12)
    ax.set_title('Domain Evolution Over Time - OPTIMIZED', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cr3bp_domain_evolution_optimized.png', dpi=300, bbox_inches='tight')


def compute_expansion_errors(
    sample_points: np.ndarray,
    assignments_opt: List[Dict],
    assignments_std: List[Dict]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute position and velocity errors between optimized and standard expansions.
    Args:
        sample_points: Array of sample points to evaluate (N x 6)
        assignments_opt: List of assignment dictionaries from OPTIMIZED method
        assignments_std: List of assignment dictionaries from STANDARD method
    Returns:
        error_pos: Array of position errors (%)
        error_vel: Array of velocity errors (%)
    """
    error_tot_pos = []
    error_tot_vel = []
    sample_finals_opt = np.zeros((len(sample_points), 6))
    sample_finals_std = np.zeros((len(sample_points), 6))

    for assignment in assignments_opt:
        # OPTIMIZED assignment
        for adimensional_point, index in zip(
            assignment["adimensional_points"], assignment["indices"]  # Fix 1: was `assignment` (iterates dict keys)
        ):
            sample_finals_opt[index] = assignment["geometry"]["DA_map"].eval(adimensional_point)
        # Fix 2: removed erroneous `break` that aborted after the first assignment

    for assignment in assignments_std:
        # STANDARD assignment
        for adimensional_point, index in zip(
            assignment["adimensional_points"], assignment["indices"]  # Fix 1 (same)
        ):
            sample_finals_std[index] = assignment["geometry"]["DA_map"].eval(adimensional_point)
        # Fix 2: removed erroneous `break` (same)

    # Fix 3: added loop over evaluated points — `expansion_opt/std` were undefined before
    for expansion_opt, expansion_std in zip(sample_finals_opt, sample_finals_std):
        error_pos = (
            np.linalg.norm(expansion_opt[0:3] - expansion_std[0:3])
            / np.linalg.norm(expansion_std[0:3])
            * 100
        )
        error_vel = (
            np.linalg.norm(expansion_opt[3:6] - expansion_std[3:6])
            / np.linalg.norm(expansion_std[3:6])
            * 100
        )
        error_tot_pos.append(error_pos)
        error_tot_vel.append(error_vel)

    return np.array(error_tot_pos), np.array(error_tot_vel)  # Fix 4: missing return
    


def compute_cumulative_distribution(errors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cumulative distribution function (CDF) of errors.
    
    Args:
        errors: Array of errors
    
    Returns:
        sorted_errors: Errors sorted in ascending order
        cumulative_percentage: Percentage of points below each error threshold
    """
    # Sort errors in ascending order
    sorted_errors = np.sort(errors)
    
    # Compute cumulative percentage (0-100%)
    n_points = len(sorted_errors)
    cumulative_percentage = np.arange(1, n_points + 1) / n_points * 100
    
    return sorted_errors, cumulative_percentage


def print_percentile_summary(error_pos: np.ndarray, error_vel: np.ndarray) -> None:
    """
    Print percentile summary of errors between OPTIMIZED and STANDARD.
    
    Args:
        error_pos: Array of position errors (%)
        error_vel: Array of velocity errors (%)
    """
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    
    print("\n" + "="*70)
    print("ERROR PERCENTILE SUMMARY (OPTIMIZED vs STANDARD)")
    print("="*70)
    
    print("\nPOSITION ERRORS:")
    print(f"{'Percentile':<15} {'Error Threshold (%)':<25} {'Interpretation'}")
    print("-" * 70)
    for pct in percentiles:
        threshold = np.percentile(error_pos, pct)
        print(f"{pct}th{'':<12} {threshold:<25.6f} {pct}% of points have error < {threshold:.6f}%")
    
    print("\nVELOCITY ERRORS:")
    print(f"{'Percentile':<15} {'Error Threshold (%)':<25} {'Interpretation'}")
    print("-" * 70)
    for pct in percentiles:
        threshold = np.percentile(error_vel, pct)
        print(f"{pct}th{'':<12} {threshold:<25.6f} {pct}% of points have error < {threshold:.6f}%")
    
    print("="*70 + "\n")


def main():
    """
    CR3BP propagation with ADS:
    - STANDARD vs OPTIMIZED integrator comparison
    - domain evolution in time
    - 3D visualization of ADS boxes (position & velocity)
    - analysis at final time to inspect domain deformation
    """

    # ======================================================================
    # 1. DA & PROBLEM SETUP
    # ======================================================================
    n_order_DA = 4
    state_dim = 6
    DA.init(n_order_DA, state_dim)
    DA.setEps(1e-16)

    mu = 0.01215  # Earth–Moon

    # L1 halo-like initial condition
    L1_x = 0.83691513
    x0, y0, z0 = L1_x + 0.01, 0.0, 0.04
    vx0, vy0, vz0 = 0.0, 0.1, 0.0

    # Uncertainties
    dx, dy, dz = 5e-6, 1e-6, 1e-6
    dvx, dvy, dvz = 1e-7, 1e-7, 1e-7

    vect_0 = np.array([x0, y0, z0, vx0, vy0, vz0])  
    domain_matrix = np.diag([dx, dy, dz, dvx, dvy, dvz])
    # DA initial state
    domain_DA = [
        vect_0[i] + sum(
            DA(k + 1) * domain_matrix[i, k]
            for k in range(state_dim)
        )
            for i in range(state_dim)
    ]

    init_domains = [ADS(domain_DA, [])]

    # ======================================================================
    # 2. TIME & ADS PARAMETERS
    # ======================================================================
    T0, TF = 0.0, 3.8
    N_steps = 20
    t_eval = np.linspace(T0, TF, N_steps)
    index_to_check = N_steps-1 # set to len N_steps-1 s for a fair computational time comparison

    toll = 1e-9
    Nmax = 30

    print("=" * 70)
    print("CR3BP – ADS PROPAGATION COMPARISON")
    print("=" * 70)
    print(f"mu = {mu}")
    print(f"Initial state: [{x0}, {y0}, {z0}, {vx0}, {vy0}, {vz0}]")
    print(f"Uncertainty: position ~1e-6, velocity ~1e-7")
    print(f"Time span: {T0} → {TF}")
    print("=" * 70)

    # ======================================================================
    # 3. STANDARD ADS PROPAGATION
    # ======================================================================
    print("\n[1/2] STANDARD ADS integrator")

    t0_std = time.time()
    prop_std = AutomaticADS_CR3BP_integrator(mu, RK.RK78())
    prop_std.loadTime(T0, t_eval[index_to_check])
    prop_std.loadTol(1e-16, 1e-16)
    prop_std.loadStepSize()
    prop_std.loadADSopt(toll, Nmax)

    listOut_std = prop_std.propagate(init_domains, T0,  t_eval[index_to_check])
    t_elapsed_std = time.time() - t0_std

    n_domains_std = len(listOut_std)
    
    print(f"✓ STANDARD - Time: {t_elapsed_std:.3f} s")
    print(f"✓ STANDARD - Final domains: {n_domains_std}")

    # ======================================================================
    # 4. OPTIMIZED ADS PROPAGATION
    # ======================================================================
    print("\n[2/2] OPTIMIZED ADS integrator")

    t0_opt = time.time()
    prop_opt = AutomaticADS_CR3BP_integrator_optimized(mu, RK.RK78())
    prop_opt.loadTime(T0, TF)
    prop_opt.loadTol(1e-16, 1e-16)
    prop_opt.loadStepSize()
    prop_opt.loadADSopt(toll, Nmax)

    listOut_opt = prop_opt.propagate(init_domains, t_eval)
    t_elapsed_opt = time.time() - t0_opt

    n_domains_opt = len(listOut_opt[-1])
    
    print(f"✓ OPTIMIZED - Time: {t_elapsed_opt:.3f} s")
    print(f"✓ OPTIMIZED - Final domains: {n_domains_opt}")
    print(f"✓ Speedup: {t_elapsed_std / t_elapsed_opt:.2f}x")

    # ======================================================================
    # 5. DOMAIN EVOLUTION (OPTIMIZED)
    # ======================================================================
    print("\nOPTIMIZED - Domain evolution:")
    for t, states in zip(t_eval, listOut_opt):
        print(f"  t = {t:6.3f} → {len(states):4d} domains")

    plot_domain_evolution(listOut_opt, t_eval)

    # ======================================================================
    # 6. SAMPLE POINTS FOR FINAL ANALYSIS
    # ======================================================================
    n_samples = 200
    sample_points = np.random.uniform(
        low=[x0-dx, y0-dy, z0-dz, vx0-dvx, vy0-dvy, vz0-dvz],
        high=[x0+dx, y0+dy, z0+dz, vx0+dvx, vy0+dvy, vz0+dvz],
        size=(n_samples, 6)
    )

    # ======================================================================
    # 7. FINAL TIME ANALYSIS - OPTIMIZED
    # ======================================================================
    print("\n" + "=" * 70)
    print("FINAL TIME DOMAIN ANALYSIS - OPTIMIZED")
    print("=" * 70)

    final_states_opt = listOut_opt[index_to_check]
    assignments_opt = ADS_utils.assign_points_to_domains(
        final_states_opt,
        sample_points,
        domain_matrix,
        vect_0,
        n_order_DA
    )

    viz_data_opt = ADS_utils.prepare_visualization_data(
        final_states_opt,
        sample_points,
        assignments_opt
    )

    print("\nOPTIMIZED - Assignment Statistics:")
    ADS_utils.print_assignment_statistics(viz_data_opt)
    print("\nOPTIMIZED - Validation Report:")
    ADS_utils.print_validation_report(viz_data_opt)

    # ======================================================================
    # 8. FINAL TIME ANALYSIS - STANDARD
    # ======================================================================
    print("\n" + "=" * 70)
    print("FINAL TIME DOMAIN ANALYSIS - STANDARD")
    print("=" * 70)

    final_states_std = listOut_std

    assignments_std = ADS_utils.assign_points_to_domains(
        final_states_std,
        sample_points,
        domain_matrix,
        vect_0,
        n_order_DA
    )

    viz_data_std = ADS_utils.prepare_visualization_data(
        final_states_std,
        sample_points,
        assignments_std
    )

    print("\nSTANDARD - Assignment Statistics:")
    ADS_utils.print_assignment_statistics(viz_data_std)
    print("\nSTANDARD - Validation Report:")
    ADS_utils.print_validation_report(viz_data_std)

    # ======================================================================
    # 9. 3D VISUALIZATION – POSITION & VELOCITY
    # ======================================================================
    print("\n" + "=" * 70)
    print("GENERATING 3D VISUALIZATIONS")
    print("=" * 70)
    
    print("\nGenerating OPTIMIZED 3D ADS box plots (final time)...")
    boxes_opt = ADS_utils.extract_ads_boxes_and_centers(
    final_states_opt,
    DA_order=n_order_DA,
    )
    plot_ADS_boxes_2D(
        boxes_opt,
        title_prefix=" standard domain evolution",
        save=False,
        x0=vect_0,
        basis_matrix=domain_matrix,
        points= sample_points,
        dpi=300,
    )

    print("Generating STANDARD 3D ADS box plots (final time)...")
    boxes_std = ADS_utils.extract_ads_boxes_and_centers(
    final_states_std,
    DA_order=n_order_DA,
    )
    plot_ADS_boxes_2D(
        boxes_std,
        title_prefix=" standard domain evolution",
        save=False,
        x0=vect_0,
        basis_matrix=domain_matrix,
        points= sample_points,
        dpi=300,
    )


    # ======================================================================
    # 10. ERROR ANALYSIS: OPTIMIZED VS STANDARD
    # ======================================================================
    print("\n" + "=" * 70)
    print("ERROR ANALYSIS: OPTIMIZED vs STANDARD")
    print("=" * 70)
    
    # Compute errors
    error_pos, error_vel = compute_expansion_errors(
        sample_points=sample_points,
        assignments_opt=assignments_opt,
        assignments_std=assignments_std
    )
    
    # Print percentile summary
    print_percentile_summary(error_pos, error_vel)
    
    # ======================================================================
    # 11. TAYLOR MAP EXTRACTION
    # ======================================================================
    print("\n" + "=" * 70)
    print("TAYLOR MAP EXTRACTION")
    print("=" * 70)
    
    print("\nExtracting Taylor maps - OPTIMIZED...")
    taylor_maps_opt = ADS_utils.extract_all_taylor_maps(final_states_opt, order_DA=n_order_DA)
    print(f"✓ OPTIMIZED - Extracted {len(taylor_maps_opt)} Taylor maps")
    
    print("\nExtracting Taylor maps - STANDARD...")
    taylor_maps_std = ADS_utils.extract_all_taylor_maps(final_states_std, order_DA=n_order_DA)
    print(f"✓ STANDARD - Extracted {len(taylor_maps_std)} Taylor maps")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  STANDARD:  {n_domains_std} domains in {t_elapsed_std:.3f} s")
    print(f"  OPTIMIZED: {n_domains_opt} domains in {t_elapsed_opt:.3f} s")
    print(f"  Speedup:   {t_elapsed_std / t_elapsed_opt:.2f}x")
    print("=" * 70)
    
    plt.show()


if __name__ == "__main__":
    main()
