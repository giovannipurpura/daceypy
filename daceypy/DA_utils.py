from __future__ import annotations
from typing import List, Dict
from itertools import permutations
from math import factorial
from collections import Counter
import numpy as np
from numpy.typing import NDArray


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


def extract_map(
    sol: List,
    max_order: int
) -> List[Dict[str, NDArray[np.float64]]]:
    """
    Extract Taylor expansion terms (0th, 1st, 2nd, ...) from a DA state transition map.

    Parameters
    ----------
    sol : list
        A list of DA state vectors (e.g., output of a DA ODE propagator),
        one entry per time instant. Each sol[j][i] is a DA polynomial
        representing the i-th state component.
    max_order : int
        Maximum Taylor expansion order to extract (≥ 0).

    Returns
    -------
    list[dict[str, NDArray[np.float64]]]
        A list of dictionaries, one per time instant.
        Keys follow the naming convention:
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
        If `max_order` is not a non-negative integer.

    Notes
    -----
    This function converts DA polynomial representations into structured
    NumPy tensors suitable for sensitivity analysis, uncertainty propagation,
    or higher-order control and estimation.
    """
    if not isinstance(max_order, int) or max_order < 0:
        raise ValueError(f"'max_order' must be an integer ≥ 0, got {max_order!r}")

    n_instants = len(sol)
    n_state = len(sol[0])

    expansion = []

    for j in range(n_instants):
        sol_j = sol[j]
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
                denom = factorial(len(multi_idx)) / np.prod([factorial(v) for v in counts.values()])
                adjusted_coeff = coeff / denom

                fill_symmetric_tensor(
                    taylor_terms[f"Taylor_order_{order}"][i],
                    multi_idx,
                    adjusted_coeff
                )

        expansion.append(taylor_terms)

    return expansion
