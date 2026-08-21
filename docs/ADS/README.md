# Adaptive Domain Splitting (ADS)

ADS is included for DA-based uncertainty propagation in nonlinear dynamics. It
can split a DA domain when truncation error thresholds indicate that a single
Taylor expansion is no longer accurate enough.

## API overview

- domain splitting and replay through `ADS`
- split direction and truncation-error checks
- box center, width, and containment helpers
- static evaluation of split domain collections
- ADS-aware propagation through `ADSintegrator` and `ADSintegrator_optimized`
- data extraction, point assignment, validation, and visualization utilities in
  `ADS_utils`

## References

Information about ADS can be found in:

`Wittig, A., Di Lizia, P., Armellin, R. et al. Propagation of large uncertainty sets in orbital dynamics by automatic domain splitting. Celest Mech Dyn Astr 122, 239–261 (2015).` https://doi.org/10.1007/s10569-015-9618-3

This is the abstract of the article:

```Current approaches to uncertainty propagation in astrodynamics mainly refer to linearized models or Monte Carlo simulations. Naive linear methods fail in nonlinear dynamics, whereas Monte Carlo simulations tend to be computationally intensive. Differential algebra has already proven to be an efficient compromise by replacing thousands of pointwise integrations of Monte Carlo runs with the fast evaluation of the arbitrary order Taylor expansion of the flow of the dynamics. However, the current implementation of the DA-based high-order uncertainty propagator fails when the non-linearities of the dynamics prohibit good convergence of the Taylor expansion in one or more directions. We solve this issue by introducing automatic domain splitting. During propagation, the polynomial expansion of the current state is split into two polynomials whenever its truncation error reaches a predefined threshold. The resulting set of polynomials accurately tracks uncertainties, even in highly nonlinear dynamics. The method is tested on the propagation of (99942) Apophis post-encounter motion.```

## Examples

- Example 1 (Basics) [Py](1Basics-Ex.py)
- Example 2 (Advanced) [Py](2Advanced-Ex.py)
- Example 3 (Online ADS propagation) [Py](3OnlineADS-Ex.py)
- Example 4 (Optimized ADS propagation and benchmark) [Py](4OptimizedADS-Ex.py)
