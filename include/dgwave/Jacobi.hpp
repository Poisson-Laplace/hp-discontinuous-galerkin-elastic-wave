#pragma once
#include "dgwave/Types.hpp"

namespace dgw::jacobi {

/// Orthonormal Jacobi polynomial P~_n^(a,b)(x) on [-1,1], i.e.
///     \int_{-1}^{1} (1-x)^a (1+x)^b P~_m P~_n dx = delta_mn.
Real P(Real x, Real a, Real b, int n);

/// d/dx of the orthonormal Jacobi polynomial.
/// Uses  d/dx P~_n^(a,b) = sqrt(n (n+a+b+1)) * P~_{n-1}^(a+1,b+1).
Real dP(Real x, Real a, Real b, int n);

/// An n-point quadrature rule on [-1,1].
struct Rule {
  Vec x;  ///< nodes, ascending
  Vec w;  ///< weights (the weight function is already folded in)
  int size() const { return static_cast<int>(x.size()); }
};

/// n-point Gauss-Jacobi rule for weight (1-x)^a (1+x)^b.
/// Exact for polynomial integrands up to degree 2n-1.
/// Built by Golub-Welsch: eigen-decomposition of the Jacobi matrix.
Rule gaussJacobi(int n, Real a, Real b);

/// n-point Gauss-Legendre rule (a = b = 0). Exact to degree 2n-1.
Rule gaussLegendre(int n);

/// Cached Gauss-Legendre rule. Used for face integration, where the same
/// rule size is requested over and over across the whole mesh.
const Rule& gaussLegendreCached(int n);

}  // namespace dgw::jacobi
