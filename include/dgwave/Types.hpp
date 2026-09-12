#pragma once
//
//  dgwave — modal hp Discontinuous Galerkin engine for elastic waves.
//
//  Locked-in design decisions (do not silently break these):
//    * velocity–stress first-order hyperbolic system  -> clean discrete adjoint
//    * orthonormal modal (Dubiner) basis on simplices -> mass matrix = J * I,
//      so no mass matrix is ever inverted, at any p
//    * per-element polynomial degree p                -> hp-ready from day one
//    * hierarchical mode ordering (graded by total degree) -> p-refinement is
//      zero-padding, p-coarsening is truncation
//    * element-wise constant material                 -> right granularity for
//      the FWI gradient
//    * RHS is a linear operator                       -> discrete adjoint is
//      exactly its transpose
//
#include <Eigen/Dense>
#include <cstddef>

namespace dgw {

using Real = double;

using Vec  = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
using Mat  = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;
using Vec2 = Eigen::Matrix<Real, 2, 1>;

/// Tight tolerance used by the verification tests and degeneracy guards.
inline constexpr Real kTiny = 1e-13;

/// Highest polynomial degree the operator caches will accept.
inline constexpr int kMaxDegree = 20;

}  // namespace dgw
