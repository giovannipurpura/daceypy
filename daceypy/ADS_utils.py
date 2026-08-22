import itertools
from typing import Any, Dict, List, Optional, Union

import numpy as np

from daceypy.DA_utils import extract_map


def _patch_corners_from_box(patch: Any) -> np.ndarray:
    """Return all physical patch corners as an array of shape (2**ndim, ndim)."""
    ndim = len(patch.manifold)
    corners = [
        np.asarray(patch.box.eval(list(signs)), dtype=float).ravel()
        for signs in itertools.product([-1, 1], repeat=ndim)
    ]
    return np.asarray(corners, dtype=float)


def _patch_axes_from_box(patch: Any) -> np.ndarray:
    """Return the patch axes matrix whose columns are the box edge vectors."""
    ndim = len(patch.manifold)
    origin = np.asarray(patch.box.eval([0] * ndim), dtype=float).ravel()
    axes = np.zeros((ndim, ndim), dtype=float)
    for k in range(ndim):
        basis_vec = [1 if j == k else 0 for j in range(ndim)]
        axes[:, k] = np.asarray(patch.box.eval(basis_vec), dtype=float).ravel() - origin
    return axes

def extract_ads_boxes_and_centers(
    ADS_domains: Union[List[Any], List[List[Any]]],
    DA_order: int,
) -> Union[List[Dict], List[List[Dict]]]:
    """
    Extract the physical ADS boxes plus their centers.
    """
    is_multiple_times = (
        isinstance(ADS_domains, list)
        and len(ADS_domains) > 0
        and isinstance(ADS_domains[0], list)
    )
    ADS_domains_list = ADS_domains if is_multiple_times else [ADS_domains]

    all_geometries: List[List[Dict]] = []

    for ADS_domains_at_time in ADS_domains_list:
        geometries_at_time: List[Dict] = []
        for domain in ADS_domains_at_time:
            patch = domain.ADSPatch
            ndim = len(patch.manifold)

            physical_origin = np.asarray(patch.box.eval([0] * ndim), dtype=float).ravel()
            physical_corners = _patch_corners_from_box(patch)

            geometries_at_time.append(
                {
                    "DA_map": patch.manifold,
                    "Taylor_map": extract_map(patch.manifold, DA_order),
                    "patch_axes": _patch_axes_from_box(patch),
                    "physical": {
                        "origin": physical_origin,
                        "corners": physical_corners,
                    },
                }
            )
        all_geometries.append(geometries_at_time)

    return all_geometries[0] if not is_multiple_times else all_geometries


def extract_all_taylor_maps(
    ADS_domains: Union[List[Any], List[List[Any]]],
    DA_order: int,
) -> Union[List[Dict], List[List[Dict]]]:
    """
    Extract Taylor maps from all ADS domains independently of any points.
    """
    is_multiple_times = (
        isinstance(ADS_domains, list)
        and len(ADS_domains) > 0
        and isinstance(ADS_domains[0], list)
    )
    ADS_domains_list = ADS_domains if is_multiple_times else [ADS_domains]

    all_taylor_maps: List[List[Dict]] = []
    for ADS_domains_at_time in ADS_domains_list:
        taylor_maps_at_time: List[Dict] = []
        for domain in ADS_domains_at_time:
            patch = domain.ADSPatch
            ndim = len(patch.manifold)
            origin = np.asarray(patch.box.eval([0] * ndim), dtype=float).ravel()

            taylor_maps_at_time.append(
                {
                    "Taylor_map": extract_map(patch.manifold, DA_order),
                    "DA_map": patch.manifold,
                    "box": {
                        "origin": origin,
                        "corners": _patch_corners_from_box(patch),
                    },
                    "manifold_dim": ndim,
                }
            )
        all_taylor_maps.append(taylor_maps_at_time)

    return all_taylor_maps[0] if not is_multiple_times else all_taylor_maps


def assign_points_to_domains(
    ADS_domains: Union[List[Any], List[List[Any]]],
    points: np.ndarray,
    domain_matrix: np.ndarray,
    origin: np.ndarray,
    DA_order: int,
) -> Union[List[Dict], List[List[Dict]]]:
    """
    Assign physical points to the corresponding ADS sub-domains or patches.

    This function determines the membership of discrete points within a set of
    partitioned Differential Algebra (DA) domains. It handles coordinate transformations
    between the global physical space and the local normalized (adimensional)
    coordinates of each patch.

    Parameters
    ----------
    ADS_domains : list or list of lists
        The set of ADS domains (potentially over multiple time steps). The
        initial box must be affine (degree 1): ``ADS.split`` preserves that
        through any sequence of splits, which is what keeps every patch an
        axis-aligned box once rotated into ``domain_matrix``'s frame. A
        higher-degree initial box would be linearized in silence by
        ``_patch_axes_from_box``.
    points : ndarray, shape (n_points, ndim)
        Physical state vectors to be assigned to sub-domains.
    domain_matrix : ndarray, shape (ndim, ndim)
        The primary transformation matrix defining the global DA domain.
        Must be the same matrix that generated ``ADS_domains``: the exact
        membership test below relies on every patch remaining an affine
        (degree-1) box in this frame, and silently degrades to an
        approximate axis-aligned check if a different matrix is passed.
    origin : ndarray, shape (ndim,)
        The global center of the parent ADS domain.
    DA_order : int
        The order of the Taylor expansions within the patches.

    Returns
    -------
    list or list of lists
        A structured assignment containing:
        - domain_index: The index of the assigned patch.
        - geometry: Local Taylor maps, axes, and physical boundaries.
        - physical_points: Points assigned to this domain in state coordinates.
        - adimensional_points: Local coordinates in the range [-1, 1]^n.

    Notes
    -----
    Every point is always assigned to *some* domain: a point that falls
    outside every patch boundary is assigned to the nearest one instead
    (nearest patch center, in the domains' rotated frame). There is
    currently no way for the caller to tell, from the return value alone,
    whether a given point was actually inside a patch or only assigned by
    this fallback -- e.g. its ``adimensional_points`` may fall outside
    ``[-1, 1]^n`` in that case.
    """
    points_array = np.asarray(points, dtype=np.float64)
    if points_array.ndim == 1:
        points_array = points_array.reshape(1, -1)
    n_points, ndim = points_array.shape

    # Check if the input contains domains for multiple time-steps
    is_multiple_times = (
        isinstance(ADS_domains, list)
        and len(ADS_domains) > 0
        and isinstance(ADS_domains[0], list)
    )
    domains_list = ADS_domains if is_multiple_times else [ADS_domains]
    inv_domain_matrix = np.linalg.inv(domain_matrix)

    all_point_assignments: List[List[Dict]] = []
    membership_tol = 1e-10

    for ADS_domains_at_time in domains_list:
        patches = [domain.ADSPatch for domain in ADS_domains_at_time]
        n_domains = len(patches)

        # Retrieve geometry data (corners, origins, and Taylor maps) for each patch
        geometries = extract_ads_boxes_and_centers(ADS_domains_at_time, DA_order)

        physical_origins = np.array([g["physical"]["origin"] for g in geometries], dtype=float)
        physical_corners = np.array([g["physical"]["corners"] for g in geometries], dtype=float)
        patch_axes = np.array([g["patch_axes"] for g in geometries], dtype=float)

        point_to_domain = np.full(n_points, -1, dtype=np.int64)

        # Transform physical points to the normalized global DA space
        rotated_points = (inv_domain_matrix @ (points_array - origin).T).T

        # Pre-compute rotated boundaries for efficient membership testing
        rotated_mins = np.empty((n_domains, ndim), dtype=float)
        rotated_maxs = np.empty((n_domains, ndim), dtype=float)
        rotated_centers = np.empty((n_domains, ndim), dtype=float)

        for d in range(n_domains):
            rotated_corners = (inv_domain_matrix @ (physical_corners[d] - origin).T).T
            rotated_mins[d] = rotated_corners.min(axis=0)
            rotated_maxs[d] = rotated_corners.max(axis=0)
            rotated_centers[d] = rotated_corners.mean(axis=0)

        # vectorized check: check if points fall within the rotated bounding box of each patch
        # (broadcasts (n_points, 1, ndim) against (1, n_domains, ndim), no need to
        # materialize the repeated (n_points, n_domains, ndim) array)
        inside_rotated_box = np.all(
            (rotated_points[:, np.newaxis, :] >= (rotated_mins[np.newaxis, :, :] - membership_tol))
            & (rotated_points[:, np.newaxis, :] <= (rotated_maxs[np.newaxis, :, :] + membership_tol)),
            axis=2,
        )

        # Spatial tree for handling points outside the explicitly defined domains,
        # built on the same rotated frame used for the membership/disambiguation
        # tests above (querying in physical space would use a different metric
        # whenever domain_matrix is anisotropic).
        # Imported lazily: scipy is an optional dependency, needed only here.
        from scipy.spatial import KDTree
        tree = KDTree(rotated_centers)

        for i in range(n_points):
            candidates = np.where(inside_rotated_box[i])[0]
            if candidates.size == 1:
                point_to_domain[i] = candidates[0]
            elif candidates.size > 1:
                # Disambiguate overlapping boundaries using distance to the patch center
                distances = [
                    np.linalg.norm(rotated_points[i] - rotated_centers[d])
                    for d in candidates
                ]
                point_to_domain[i] = candidates[int(np.argmin(distances))]
            else:
                # Assign to the nearest domain if no boundary is intersected.
                # Deliberate: this always returns a domain, even for a point
                # arbitrarily far from every one of them -- there is no
                # sentinel/warning for "unassigned" (see the Notes section
                # of this function's docstring).
                _, nearest = tree.query(rotated_points[i])
                point_to_domain[i] = nearest

        # Compile assignment results with local adimensional coordinates
        point_assignments: List[Dict] = []
        for d in range(n_domains):
            idx = np.where(point_to_domain == d)[0]
            if idx.size == 0:
                continue

            physical_points = points_array[idx]
            # Convert physical coordinates back to local patch space [-1, 1]
            adimensional_points = np.array(
                [np.linalg.solve(patch_axes[d], x - physical_origins[d]) for x in physical_points],
                dtype=float,
            )

            entry: Dict = {
                "domain_index": d,
                "n_points": len(idx),
                "geometry": {
                    "DA_map": patches[d].manifold,
                    "Taylor_map": geometries[d]["Taylor_map"],
                    "patch_axes": patch_axes[d],
                    "physical": {
                        "origin": physical_origins[d],
                        "corners": physical_corners[d],
                    },
                },
                'indices': idx,
                'point_indices': idx,
                "physical_points": physical_points,
                "adimensional_points": adimensional_points,
            }
            point_assignments.append(entry)

        all_point_assignments.append(point_assignments)

    return all_point_assignments[0] if not is_multiple_times else all_point_assignments

def prepare_visualization_data(
    ADS_domains: List[Any],
    points: np.ndarray,
    point_assignments: Optional[List[Dict]] = None
) -> Dict:
    """
    Prepare data structures for visualization and analysis.

    Parameters
    ----------
    ADS_domains : list[ADSstate]
        List of ADSstate objects (with ADSPatch attribute)
    points : np.ndarray of shape (N, ndim)
        Points that were assigned to domains
    point_assignments : list[dict], optional
        Output from assign_points_to_domains. If None, points will be assigned automatically.

    Returns
    -------
    visualization_data : dict
        Dictionary containing:
            - 'points': np.ndarray - the original points array
            - 'domain_boxes': list[dict] - bounding boxes for each domain
            - 'point_to_domain': np.ndarray - mapping each point to its assigned domain index
            - 'n_domains': int - total number of domains
            - 'n_points': int - total number of points
    """
    # Convert points to numpy array and validate dimensions
    points_array: np.ndarray = np.asarray(points, dtype=np.float64)
    if points_array.ndim == 1:
        points_array = points_array.reshape(1, -1)

    n_points: int
    ndim: int
    n_points, ndim = points_array.shape

    # Extract ADS patches from domain objects
    patches: List[Any] = [domain.ADSPatch for domain in ADS_domains]
    n_domains: int = len(patches)

    # Compute bounding boxes for all domains
    domain_boxes: List[Dict] = []
    for patch in patches:
        # Use all 2**ndim corners, not just the (-1,...,-1)/(+1,...,+1)
        # diagonal: for a rotated domain_matrix the axis-wise min/max of the
        # box is not generally attained at those two corners alone.
        corners = _patch_corners_from_box(patch)
        min_corner: np.ndarray = corners.min(axis=0)
        max_corner: np.ndarray = corners.max(axis=0)

        domain_boxes.append({
            "min": min_corner,
            "max": max_corner,
            "center": 0.5 * (min_corner + max_corner),
            "manifold": patch.manifold
        })

    # Build point-to-domain mapping
    point_to_domain: np.ndarray

    if point_assignments is None:
        # Compute assignments if not provided
        point_to_domain = np.full(n_points, -1, dtype=np.int64)

        # Membership must be tested against each patch's own (possibly
        # rotated) parallelepiped, not its axis-aligned bounding box: for a
        # rotated domain_matrix the AABBs of neighboring patches overlap
        # even where the true patches do not, which would misassign points
        # in that overlap region.
        membership_tol = 1e-10
        patch_axes = [_patch_axes_from_box(patch) for patch in patches]
        patch_origins = [
            np.asarray(patch.box.eval([0] * ndim), dtype=float).ravel()
            for patch in patches
        ]

        for i, point in enumerate(points_array):
            is_inside: List[bool] = [
                np.all(np.abs(np.linalg.solve(patch_axes[d], point - patch_origins[d])) <= 1.0 + membership_tol)
                for d in range(n_domains)
            ]

            if any(is_inside):
                candidate_indices: np.ndarray = np.where(is_inside)[0]
                distances: List[float] = [
                    np.linalg.norm(point - domain_boxes[idx]["center"])
                    for idx in candidate_indices
                ]
                point_to_domain[i] = candidate_indices[np.argmin(distances)]
            else:
                distances = [
                    np.linalg.norm(point - box["center"])
                    for box in domain_boxes
                ]
                point_to_domain[i] = np.argmin(distances)
    else:
        # Reconstruct from assignments
        point_to_domain = np.full(n_points, -1, dtype=np.int64)
        for assignment in point_assignments:
            domain_idx: int = assignment['domain_index']
            point_indices: np.ndarray = assignment.get('point_indices', assignment.get('indices'))
            if point_indices is None:
                raise KeyError("point assignment must contain 'point_indices' or 'indices'")
            point_to_domain[point_indices] = domain_idx

    # Prepare visualization data
    visualization_data: Dict = {
        "points": points_array,
        "domain_boxes": domain_boxes,
        "point_to_domain": point_to_domain,
        "n_domains": n_domains,
        "n_points": n_points
    }

    return visualization_data


def compute_assignment_statistics(visualization_data: Dict) -> Dict:
    """
    Calculate statistics about point-to-domain assignments.

    Parameters
    ----------
    visualization_data : dict
        Output from prepare_visualization_data

    Returns
    -------
    stats : dict
        Dictionary containing:
            - 'n_domains': int - total number of domains
            - 'n_points': int - total number of points
            - 'points_per_domain': np.ndarray - point counts per domain
            - 'empty_domains': int - number of domains with no assigned points
            - 'occupied_domains': int - number of domains with at least one point
            - 'mean_points_per_occupied_domain': float - average points per non-empty domain
            - 'min_points_per_occupied_domain': int - minimum points in non-empty domains
            - 'max_points_per_domain': int - maximum points in any domain
            - 'coverage_ratio': float - fraction of domains with assigned points
    """
    point_to_domain: np.ndarray = visualization_data['point_to_domain']
    n_domains: int = visualization_data['n_domains']
    n_points: int = visualization_data['n_points']

    # Count points per domain
    points_per_domain: np.ndarray = np.array([
        np.sum(point_to_domain == domain_idx)
        for domain_idx in range(n_domains)
    ], dtype=np.int64)

    # Calculate statistics
    empty_domains: int = int(np.sum(points_per_domain == 0))
    occupied_domains: int = n_domains - empty_domains

    occupied_counts: np.ndarray = points_per_domain[points_per_domain > 0]
    mean_points_occupied: float = float(np.mean(occupied_counts)) if occupied_domains > 0 else 0.0
    min_points_occupied: int = int(np.min(occupied_counts)) if occupied_domains > 0 else 0
    max_points: int = int(np.max(points_per_domain))

    stats: Dict = {
        'n_domains': n_domains,
        'n_points': n_points,
        'points_per_domain': points_per_domain,
        'empty_domains': empty_domains,
        'occupied_domains': occupied_domains,
        'mean_points_per_occupied_domain': mean_points_occupied,
        'min_points_per_occupied_domain': min_points_occupied,
        'max_points_per_domain': max_points,
        'coverage_ratio': float(occupied_domains) / float(n_domains) if n_domains > 0 else 0.0
    }

    return stats


def print_assignment_statistics(visualization_data: Dict) -> None:
    """
    Print formatted statistics about point-to-domain assignments.

    Parameters
    ----------
    visualization_data : dict
        Output from prepare_visualization_data

    Returns
    -------
    None
    """
    stats: Dict = compute_assignment_statistics(visualization_data)

    print("\n" + "="*70)
    print("Point-to-Domain Assignment Statistics")
    print("="*70)
    print(f"Total domains:                    {stats['n_domains']:6d}")
    print(f"Total points:                     {stats['n_points']:6d}")
    print(f"Empty domains:                    {stats['empty_domains']:6d}")
    print(f"Occupied domains:                 {stats['occupied_domains']:6d}")
    print(f"Coverage ratio:                   {stats['coverage_ratio']:6.2%}")
    print(f"Mean points/occupied domain:      {stats['mean_points_per_occupied_domain']:6.2f}")
    print(f"Min points (occupied domains):    {stats['min_points_per_occupied_domain']:6d}")
    print(f"Max points (any domain):          {stats['max_points_per_domain']:6d}")
    print("="*70)

    # Show distribution histogram
    points_per_domain: np.ndarray = stats['points_per_domain']
    unique_counts: np.ndarray
    count_frequencies: np.ndarray
    unique_counts, count_frequencies = np.unique(points_per_domain, return_counts=True)

    print("\nDistribution of points per domain:")
    print("-" * 70)
    max_freq: int = int(max(count_frequencies))
    for count, freq in zip(unique_counts, count_frequencies):
        bar: str = '█' * int(freq / max_freq * 40)
        if count == 0:
            print(f"  {freq:4d} domains with    0 points  {bar}")
        else:
            print(f"  {freq:4d} domains with {count:4d} points  {bar}")
    print("="*70 + "\n")


def validate_point_assignments(
    visualization_data: Dict,
    tolerance: float = 1e-10
) -> Dict:
    """
    Validate that all points are properly assigned to domains.

    Parameters
    ----------
    visualization_data : dict
        Output from prepare_visualization_data
    tolerance : float, default=1e-10
        Tolerance for checking if point is inside domain bounding box

    Returns
    -------
    validation : dict
        Dictionary containing:
            - 'all_assigned': bool - True if all points have valid domain assignments
            - 'n_unassigned': int - number of points with invalid assignments
            - 'n_outside_box': int - number of points outside their assigned domain box
            - 'total_points': int - total number of points
            - 'is_valid': bool - overall validation status
    """
    points: np.ndarray = visualization_data['points']
    point_to_domain: np.ndarray = visualization_data['point_to_domain']
    domain_boxes: List[Dict] = visualization_data['domain_boxes']

    n_unassigned: int = int(np.sum(point_to_domain == -1))
    n_outside_box: int = 0

    for i, (point, domain_idx) in enumerate(zip(points, point_to_domain)):
        if 0 <= domain_idx < len(domain_boxes):
            box: Dict = domain_boxes[domain_idx]
            # Check if point is inside assigned domain box (with tolerance)
            is_inside: bool = bool(np.all(
                (point >= box["min"] - tolerance) &
                (point <= box["max"] + tolerance)
            ))
            if not is_inside:
                n_outside_box += 1

    validation: Dict = {
        'all_assigned': n_unassigned == 0,
        'n_unassigned': n_unassigned,
        'n_outside_box': n_outside_box,
        'total_points': len(points),
        'is_valid': (n_unassigned == 0) and (n_outside_box == 0)
    }

    return validation


def print_validation_report(
    visualization_data: Dict,
    tolerance: float = 1e-10
) -> None:
    """
    Print a validation report for point-to-domain assignments.

    Parameters
    ----------
    visualization_data : dict
        Output from prepare_visualization_data
    tolerance : float, default=1e-10
        Tolerance for validation checks

    Returns
    -------
    None
    """
    validation: Dict = validate_point_assignments(visualization_data, tolerance)

    print("\n" + "="*70)
    print("Assignment Validation Report")
    print("="*70)
    print(f"Total points:                     {validation['total_points']:6d}")
    print(f"Unassigned points:                {validation['n_unassigned']:6d}")
    print(f"Points outside assigned box:      {validation['n_outside_box']:6d}")
    print(f"All points assigned:              {'✓ Yes' if validation['all_assigned'] else '✗ No'}")
    print(f"Overall validation:               {'✓ PASS' if validation['is_valid'] else '✗ FAIL'}")
    print("="*70 + "\n")
