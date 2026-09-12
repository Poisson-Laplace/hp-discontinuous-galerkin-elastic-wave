#pragma once
#include <vector>

#include "dgwave/Types.hpp"

namespace dgw::dubiner {

//  Reference triangle
//      T = { (r,s) : r >= -1, s >= -1, r + s <= 0 },   |T| = 2
//
//      v2 (-1, 1)
//        o
//        |  .
//    f2  |     .   f1
//        |        .
//        o---------o
//      v0 (-1,-1)   v1 (1,-1)
//             f0
//
//  Vertices are ordered counter-clockwise, so the face normals below are
//  outward-pointing.
//
//  Collapsed (Duffy) coordinates:  a = 2(1+r)/(1-s) - 1,   b = s
//
//  Basis:  psi_{ij}(r,s) = sqrt(2) * P~_i^(0,0)(a) * P~_j^(2i+1,0)(b) * (1-b)^i
//  which is L2-ORTHONORMAL on T. Hence the reference mass matrix is the
//  identity and the physical mass matrix on an affine element is J * I.

/// Number of modes for a total-degree-p space: (p+1)(p+2)/2.
constexpr int numModes(int p) { return (p + 1) * (p + 2) / 2; }

/// Tensor indices of one mode. Total polynomial degree is i + j.
struct ModeIdx {
  int i;
  int j;
};

/// Hierarchical mode list, graded by total degree:
///     d = 0..p, then i = 0..d with j = d - i.
///
/// KEY PROPERTY: modes(p) is a strict PREFIX of modes(p+1). A degree-p
/// coefficient vector is therefore the leading block of a degree-(p+1) one,
/// which makes p-refinement zero-padding and p-coarsening truncation.
/// Do not reorder this without revisiting every hp projection.
const std::vector<ModeIdx>& modes(int p);

/// (r,s) -> collapsed (a,b). The a-coordinate is singular at the collapsed
/// vertex s = 1; it is clamped there (the value is irrelevant because every
/// basis function that depends on a carries a vanishing (1-b)^i factor).
void rsToab(Real r, Real s, Real& a, Real& b);

/// Evaluate psi_{ij} at (r,s).
Real eval(Real r, Real s, int i, int j);

/// Gradient of psi_{ij} with respect to the reference coordinates (r,s).
void grad(Real r, Real s, int i, int j, Real& dpsi_dr, Real& dpsi_ds);

}  // namespace dgw::dubiner
