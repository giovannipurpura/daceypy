from __future__ import annotations
from typing import List, Dict, Union, Optional, Sequence
from itertools import permutations, combinations_with_replacement
from math import factorial
from collections import Counter
import numpy as np
from numpy.typing import NDArray
from functools import reduce

def fill_symmetric_tensor(
    tensor: NDArray[np.float64],
    indices: List[int],
    value: float
) -> None:
    """
    Fill all permutation-equivalent entries of a symmetric tensor.

    Parameters
    ----------
    tensor : NDArray[np.float64]
        The tensor to update. It is assumed to be symmetric in all indices.
    indices : list[int]
        Indices describing the monomial exponents (e.g. [0, 2, 2]).
        All unique permutations correspond to equal entries in the tensor.
    value : float
        The coefficient to assign to all symmetric tensor locations.

    Notes
    -----
    Because higher-order DA Taylor terms are symmetric with respect to
    variable permutations, this function ensures the coefficient is written
    consistently across all equivalent tensor positions.
    """
    for perm in set(permutations(indices)):
        tensor[perm] = value


# ---------------------------------------------------------------------------
# Input validation: determine if sol is a single DA vector or a list thereof
# ---------------------------------------------------------------------------
def _is_da_element(obj) -> bool:
    """Return True if obj looks like a single DA polynomial (has .cons method)."""
    return hasattr(obj, 'cons') and callable(getattr(obj, 'cons', None))

def _is_da_vector(obj) -> bool:
    """Return True if obj is an iterable of DA polynomials (a DA state vector)."""
    try:
        return (
            hasattr(obj, '__len__')
            and len(obj) > 0
            and _is_da_element(obj[0])
        )
    except (TypeError, KeyError):
        return False

def _is_list_of_da_vectors(obj) -> bool:
    """Return True if obj is a non-empty list/tuple of DA state vectors."""
    try:
        return (
            isinstance(obj, (list, tuple))
            and len(obj) > 0
            and _is_da_vector(obj[0])
        )
    except (TypeError, KeyError):
        return False

def extract_map(
    sol,
    max_order: Optional[int] = None
) -> Union[Dict[str, NDArray[np.float64]], List[Dict[str, NDArray[np.float64]]]]:
    """
    Extract Taylor expansion terms (0th, 1st, 2nd, ...) from a DA state transition map.
    
    Parameters
    ----------
    sol : DA array or list of DA arrays
        - If DA array: Single DA state vector
        - If list of DA arrays: List of DA state vectors (one per time instant)
        Each sol[i] (or sol[j][i]) is a DA polynomial representing the i-th state component.
    max_order : int, optional
        Maximum Taylor expansion order to extract (≥ 0).
        If None (default), all available orders are extracted automatically
        by inspecting the DA polynomials' monomial structure.
    
    Returns
    -------
    dict or list[dict]
        - If single DA array: Single dictionary of Taylor terms
        - If list of DA arrays: List of dictionaries (one per time instant)
        
        Each dictionary has keys following the naming convention:
            "Taylor_order_0" → constant term (state at nominal IC)
            "Taylor_order_1" → Jacobian (STM)
            "Taylor_order_2" → Hessian tensor
            ...
        Each term is stored as a NumPy array with shape:
            order = 0 → (n_state,)
            order = 1 → (n_state, n_state)
            order = 2 → (n_state, n_state, n_state)
            etc.
    
    Raises
    ------
    ValueError
        If `max_order` is provided but is not a non-negative integer.
    TypeError
        If `sol` is not a DA array or list of DA arrays.
    
    Notes
    -----
    This function converts DA polynomial representations into structured
    NumPy tensors suitable for sensitivity analysis, uncertainty propagation,
    or higher-order control and estimation.
    When `max_order` is None, the maximum order is inferred from the highest-degree
    monomial found across all state components of the first time instant.

    Examples
    --------
    >>> # Single state vector — extract all available orders
    >>> taylor_terms = extract_map(state_vector)
    >>> print(taylor_terms['Taylor_order_0'].shape)  # (n_state,)
    
    >>> # Single state vector — cap at order 2
    >>> taylor_terms = extract_map(state_vector, max_order=2)

    >>> # Multiple time instants
    >>> taylor_series = extract_map([state_0, state_1, state_2], max_order=2)
    >>> print(len(taylor_series))  # 3
    >>> print(taylor_series[0]['Taylor_order_1'].shape)  # (n_state, n_state)
    """
    if max_order is not None and (not isinstance(max_order, int) or max_order < 0):
        raise ValueError(f"'max_order' must be an integer ≥ 0, got {max_order!r}")
    
    if _is_da_vector(sol) and not _is_list_of_da_vectors(sol):
        sol_list = [sol]
        return_single = True
    elif _is_list_of_da_vectors(sol):
        sol_list = list(sol)
        return_single = False
    else:
        raise TypeError(
            "Input 'sol' must be a DA state vector or a list/tuple of DA state vectors.\n"
            f"  Expected: iterable of DA polynomials, or list thereof.\n"
            f"  Got:      {type(sol).__name__!r}"
            + (f" with first element of type {type(sol[0]).__name__!r}"
               if hasattr(sol, '__len__') and len(sol) > 0 else "")
        )

    # --- Infer max_order from the DA structure if not provided ---
    if max_order is None:
        max_order = _infer_max_order(sol_list[-1])

    # Process each time instant
    expansion = []
    
    for j, sol_j in enumerate(sol_list):
        n_state = len(sol_j)
        taylor_terms = {}
        
        # Pre-allocate tensors for each Taylor order
        for order in range(max_order + 1):
            shape = (n_state,) + (n_state,) * order
            taylor_terms[f"Taylor_order_{order}"] = np.zeros(shape)
        
        # 0th-order: nominal state (constant term)
        taylor_terms["Taylor_order_0"] = sol_j.cons()
        
        # Loop over each state component and extract monomial derivatives
        for i in range(n_state):
            n_monomials = sol_j[i].m_index.len + 1
            
            for k in range(n_monomials):
                monomial = sol_j[i].getMonomial(k)
                m_jj = np.array(monomial.m_jj, dtype=int)
                order = int(np.sum(m_jj))
                
                if order == 0 or order > max_order:
                    continue
                
                coeff = float(monomial.m_coeff.value)
                
                # Create list of repeated indices, e.g. [0, 2, 2]
                multi_idx = [idx for idx, exp in enumerate(m_jj) for _ in range(exp)]
                
                # Correct for repeated permutations (multinomial symmetry)
                counts = Counter(multi_idx)
                denom = factorial(len(multi_idx)) / np.prod(
                    [factorial(v) for v in counts.values()]
                )
                adjusted_coeff = coeff / denom
                
                fill_symmetric_tensor(
                    taylor_terms[f"Taylor_order_{order}"][i],
                    multi_idx,
                    adjusted_coeff
                )
        
        expansion.append(taylor_terms)
    
    # Return in appropriate format
    return expansion[0] if return_single else expansion


def _infer_max_order(sol_j) -> int:
    """Infer the maximum monomial order present in a DA state vector."""
    max_ord = 0
    for i in range(len(sol_j)):
        n_monomials = sol_j[i].m_index.len + 1
        for k in range(n_monomials):
            monomial = sol_j[i].getMonomial(k)
            order = int(np.sum(np.array(monomial.m_jj, dtype=int)))
            if order > max_ord:
                max_ord = order
    return max_ord


def assign_taylor_to_da(
    taylor_maps: Union[Dict[str, np.ndarray], List[Dict[str, np.ndarray]]],
    da_vars,
):
    """
    Write the Taylor tensor values back into the monomial coefficients of
    `da_vars`, in place. This mirrors `extract_map`'s monomial walk exactly,
    but assigns `monomial.m_coeff.value` instead of reading it — no new
    monomials are created and no algebra (multiplication of da_vars) is
    performed; only existing coefficients are overwritten.

    Parameters
    ----------
    taylor_maps : dict or list[dict]
        Output of `extract_map`.
    da_vars : DA state vector or list of DA state vectors
        The DA object(s) whose monomial coefficients will be overwritten.
        Must already have the same monomial support (order, variables) as
        the `sol` originally passed to `extract_map` — typically you pass
        `sol` itself (or a copy of it) here.

    Returns
    -------
    The same `da_vars` object(s), mutated in place.
    """
    is_list = isinstance(taylor_maps, list)
    maps_list = taylor_maps if is_list else [taylor_maps]

    if is_list:
        if not _is_list_of_da_vectors(da_vars) or len(da_vars) != len(maps_list):
            raise ValueError(
                "'da_vars' must be a list of DA vectors matching the length "
                "of 'taylor_maps' when the latter is a list."
            )
        da_vars_list = da_vars
    else:
        da_vars_list = [da_vars]

    for tm, sol_j in zip(maps_list, da_vars_list):
        _assign_single(tm, sol_j)

    return da_vars if is_list else da_vars_list[0]


def _assign_single(taylor_terms: Dict[str, np.ndarray], sol_j) -> None:
    """Overwrite monomial coefficients of a single DA vector `sol_j` in place."""
    max_order = max(
        int(k.rsplit("_", 1)[-1])
        for k in taylor_terms
        if k.startswith("Taylor_order_")
    )
    n_state = len(sol_j)

    for i in range(n_state):
        n_monomials = sol_j[i].m_index.len + 1

        for k in range(n_monomials):
            monomial = sol_j[i].getMonomial(k)
            m_jj = np.array(monomial.m_jj, dtype=int)
            order = int(np.sum(m_jj))

            if order > max_order:
                continue

            if order == 0:
                new_coeff = float(taylor_terms["Taylor_order_0"][i])
            else:
                multi_idx = tuple(
                    idx for idx, exp in enumerate(m_jj) for _ in range(exp)
                )
                T = taylor_terms[f"Taylor_order_{order}"][i]
                adjusted_coeff = float(T[multi_idx])

                # Undo the multinomial normalization applied in extract_map
                counts = Counter(multi_idx)
                multiplicity = factorial(order) / np.prod(
                    [factorial(c) for c in counts.values()]
                )
                new_coeff = adjusted_coeff * multiplicity

            # Only the coefficient changes — exponents/structure untouched.
            monomial.m_coeff.value = new_coeff