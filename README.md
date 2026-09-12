# dgwave

An experimental high-order modal *hp*-Discontinuous Galerkin solver for elastic wave propagation, written in C++ with inversion applications in mind.

The project started from an immersed-boundary / complex-geometry problem encountered while working on DAS-VSP full-waveform inversion. Rather than retrofitting an existing acoustic solver, I am building the numerical engine from the discretization upward, with elastic wave propagation, *hp*-adaptivity, and eventual adjoint-based inversion as core design constraints.

> **Current status:** Research prototype. The reference-element layer, modal Dubiner basis, quadrature, unstructured mesh infrastructure, exact Riemann flux assembly, semi-discrete elastic operator, and associated numerical QC suites are implemented and verified. Time integration and full waveform propagation pipelines are actively being developed.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

If Eigen is not installed system-wide:

```bash
cmake -S . -B build \
  -DDGWAVE_EIGEN_INCLUDE_DIR=/path/to/eigen3
```

## Design Decisions

These choices are intentionally coupled: changing one may require revisiting operators and assumptions elsewhere in the solver.

| Decision | Motivation |
|---|---|
| Velocity–stress first-order hyperbolic formulation | Natural P/S-wave decomposition and a clean route to numerical Riemann fluxes and adjoint formulations |
| Orthonormal modal **Dubiner** basis on triangles | Diagonal reference mass structure and efficient high-order operations on affine elements (`J·I` mass matrix) |
| Hierarchical total-degree ordering | Hierarchical coefficient transfer under p-refinement/coarsening reduces to zero-padding or truncation; inter-element fluxes are still evaluated through the corresponding face traces |
| Per-element polynomial degree | *hp*-adaptivity is part of the architecture from day one, rather than bolted on later |
| Element-wise material representation | Simple local parameterization directly compatible with FWI sensitivity kernels |
| Operator-oriented linear semi-discrete formulation | The spatial semi-discrete operator is organized so that its transpose can be used directly in a later discrete-adjoint implementation |

## Layout

```
include/dgwave/
  Types.hpp        Scalar/matrix aliases, shared constants
  Jacobi.hpp       Orthonormal Jacobi polynomials, Gauss–Jacobi quadrature
  Dubiner.hpp      2D orthonormal modal basis on the reference triangle
  RefTriangle.hpp  Cached quadrature, Vandermonde matrices and face lift operators
  Mesh.hpp         Unstructured triangular mesh with per-element polynomial degree
  Elastic.hpp      Semi-discrete elastic operator and exact Riemann solver
  Output.hpp       High-order lattice sampling and VTK writer
src/               Implementations
tests/             Verification suite (reference element, mesh, elastic operator)
tools/             Lightweight visualization utilities (quicklook.py)
```

## Conventions

Reference triangle $T = \{ (r,s) : r \ge -1, s \ge -1, r+s \le 0 \}$, area $|T| = 2$, vertices ordered counter-clockwise.

| Face | Vertices | $(r,s)(x)$ | Outward $\hat{n}$ | $\|d(r,s)/dx\|$ |
|---|---|---|---|---|
| 0 | v0 $\to$ v1 | $( x, -1)$ | $( 0,-1)$ | $1$ |
| 1 | v1 $\to$ v2 | $(-x,  x)$ | $( 1, 1)/\sqrt{2}$ | $\sqrt{2}$ |
| 2 | v2 $\to$ v0 | $(-1, -x)$ | $(-1, 0)$ | $1$ |

`faceLift` integrates in the face parameter $x \in [-1,1]$, so the physical surface term is `(physicalEdgeLength / 2) * faceLift(f, nfq) * fluxValues`, and the assembled residual is divided by the element Jacobian $J$.

Neighbours traverse a shared face in opposite directions. Gauss–Legendre nodes are symmetric, so matching them is an exact index flip (`FaceGeom::mirrorIndex`) — never an interpolation.

## Verification

The test suite pins down identities the DG residual silently depends on, rather than smoke tests:

1. **1D Jacobi orthonormality:** $\langle P_m, P_n \rangle_{(1,0)} = \delta_{mn}$, degrees 0–12.
2. **Gauss–Jacobi exactness:** Exact integration up to degree $2n-1$ in both directions.
3. **Reference triangle metrics:** $\sum w = |T| = 2$, and $V^T W V = I$ (orthonormal mass matrix) for $p = 0 \dots 10$.
4. **Analytic gradients:** $d\psi/dr, d\psi/ds$ matched against central finite differences.
5. **Divergence theorem on reference & physical elements:** $\int_T \nabla \psi \, dT = \oint_{\partial T} \psi \hat{n} \, ds$ for $p = 0 \dots 8$, verifying volume gradients, face quadrature, surface normals, and metric Jacobians simultaneously.
6. **Modal hierarchy:** `modes(p)` is a strict prefix of `modes(p+1)`; leading block of $V(p+1)$ equals the degree-$p$ basis.
7. **$L^2$ projection:** Volume reconstruction of polynomial fields and exact match on all three face traces.
8. **Eigenstructure & Riemann consistency:** Eigenvalues $\{-v_p, -v_s, 0, +v_s, +v_p\}$ verified in arbitrary directions; exact P/S wave transmission and reflection coefficients checked at material interfaces.
9. **Semi-discrete energy rate identity:** $\frac{dE}{dt} = 0$ for centered flux with energy-conserving boundaries; strictly dissipative $\frac{dE}{dt} < 0$ for Godunov upwind flux.

## Implementation Notes

- Operator caches (`refTri`, `gaussLegendreCached`) are not thread-safe during first initialization. Warm them before entering parallel regions.
- `overIntegration` defaults to `1`, making the volume rule exact through degree $2p+2$; `0` is sufficient for the exact affine-element mass matrix.
