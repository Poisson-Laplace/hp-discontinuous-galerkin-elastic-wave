#pragma once
#include <array>
#include <map>
#include <utility>

#include "dgwave/Dubiner.hpp"
#include "dgwave/Types.hpp"

namespace dgw {

/// Geometry of the three faces of the reference triangle.
///
///   f0 : v0(-1,-1) -> v1( 1,-1)   (r,s) = ( x, -1)   n = ( 0,-1)   |d(r,s)/dx| = 1
///   f1 : v1( 1,-1) -> v2(-1, 1)   (r,s) = (-x,  x)   n = ( 1, 1)/V2 |d(r,s)/dx| = V2
///   f2 : v2(-1, 1) -> v0(-1,-1)   (r,s) = (-1, -x)   n = (-1, 0)   |d(r,s)/dx| = 1
///
/// The face parameter is x in [-1,1] and the vertex order is counter-clockwise,
/// so the normals above are outward.
struct FaceGeom {
  static constexpr int kNumFaces = 3;

  /// Map a face parameter to reference coordinates.
  static void param(int face, Real x, Real& r, Real& s);

  /// Outward unit normal in reference (r,s) coordinates.
  static Vec2 refNormal(int face);

  /// |d(r,s)/dx| on the face. Needed only for reference-element identities
  /// (e.g. the divergence-theorem check); physical assembly uses the physical
  /// edge length instead — see RefTriangle::faceLift.
  static Real refSurfJac(int face);

  /// Index of the quadrature point that a neighbouring element sees at the
  /// same physical location. Faces are traversed in opposite directions from
  /// the two sides; Gauss-Legendre nodes are symmetric, so this is an exact
  /// index flip and needs no interpolation.
  static int mirrorIndex(int numFaceQuad, int q) { return numFaceQuad - 1 - q; }
};

/// Cached reference-element operators for one polynomial degree.
///
/// All operators map between MODAL coefficients and QUADRATURE point values.
/// The mass matrix is never formed or inverted: with the orthonormal Dubiner
/// basis it is exactly J * I on an affine element, so the assembled residual
/// is simply divided by the element Jacobian at the end.
class RefTriangle {
 public:
  int degree() const { return p_; }
  int numModes() const { return nm_; }
  int numVolQuad() const { return nq_; }

  /// Volume quadrature on the reference triangle. Sum of weights is |T| = 2.
  const Vec& rq() const { return rq_; }
  const Vec& sq() const { return sq_; }
  const Vec& wq() const { return wq_; }

  /// nq x nm. Modal coefficients -> values at volume quadrature points.
  const Mat& V() const { return V_; }
  /// nq x nm. Reference-coordinate derivatives at volume quadrature points.
  const Mat& Vr() const { return Vr_; }
  const Mat& Vs() const { return Vs_; }

  /// nm x nq. Weighted transposes: Wr = Vr^T diag(wq), Ws = Vs^T diag(wq).
  /// The weak volume term for mode i is  (Wr * Fr + Ws * Fs)[i], where Fr, Fs
  /// are the reference-contravariant fluxes sampled at quadrature points.
  const Mat& Wr() const { return Wr_; }
  const Mat& Ws() const { return Ws_; }

  /// nfq x nm. Modal coefficients -> values at the face quadrature points of
  /// `face`. The rule size is a free parameter so that two neighbours with
  /// different p can share one face rule (take nfq from max(pL, pR)).
  const Mat& faceV(int face, int numFaceQuad) const;

  /// nm x nfq, equal to faceV^T diag(w). Integration is with respect to the
  /// face PARAMETER x in [-1,1], so the physical surface term is
  ///     (physicalEdgeLength / 2) * faceLift(f, nfq) * fluxValues.
  const Mat& faceLift(int face, int numFaceQuad) const;

 private:
  RefTriangle(int p, int overIntegration);
  friend const RefTriangle& refTri(int, int);

  int p_ = 0;
  int nm_ = 0;
  int nq_ = 0;
  int overInt_ = 0;

  Vec rq_, sq_, wq_;
  Mat V_, Vr_, Vs_, Wr_, Ws_;

  mutable std::map<std::pair<int, int>, Mat> faceV_;
  mutable std::map<std::pair<int, int>, Mat> faceLift_;
};

/// Cached accessor. `overIntegration` adds extra quadrature points beyond the
/// minimum needed for an exact mass matrix; the default of 1 keeps the volume
/// rule exact to degree 2p+2, which absorbs mildly variable coefficients.
/// Not thread-safe on first touch — warm the cache before going parallel.
const RefTriangle& refTri(int p, int overIntegration = 1);

}  // namespace dgw
