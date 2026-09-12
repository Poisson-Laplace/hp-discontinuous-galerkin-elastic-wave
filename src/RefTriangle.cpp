#include "dgwave/RefTriangle.hpp"

#include <cmath>
#include <stdexcept>

#include "dgwave/Jacobi.hpp"

namespace dgw {

// ---------------------------------------------------------------------------
// FaceGeom
// ---------------------------------------------------------------------------

void FaceGeom::param(int face, Real x, Real& r, Real& s) {
  switch (face) {
    case 0: r = x;    s = -1.0; return;
    case 1: r = -x;   s = x;    return;
    case 2: r = -1.0; s = -x;   return;
    default: throw std::invalid_argument("FaceGeom::param: face must be 0..2");
  }
}

Vec2 FaceGeom::refNormal(int face) {
  static const Real inv_sqrt2 = 1.0 / std::sqrt(2.0);
  switch (face) {
    case 0: return Vec2(0.0, -1.0);
    case 1: return Vec2(inv_sqrt2, inv_sqrt2);
    case 2: return Vec2(-1.0, 0.0);
    default: throw std::invalid_argument("FaceGeom::refNormal: face must be 0..2");
  }
}

Real FaceGeom::refSurfJac(int face) {
  switch (face) {
    case 0: return 1.0;
    case 1: return std::sqrt(2.0);
    case 2: return 1.0;
    default: throw std::invalid_argument("FaceGeom::refSurfJac: face must be 0..2");
  }
}

// ---------------------------------------------------------------------------
// RefTriangle
// ---------------------------------------------------------------------------

RefTriangle::RefTriangle(int p, int overIntegration)
    : p_(p), nm_(dubiner::numModes(p)), overInt_(overIntegration) {
  if (p < 0 || p > kMaxDegree) throw std::invalid_argument("refTri: bad degree");
  if (overIntegration < 0) throw std::invalid_argument("refTri: bad overIntegration");

  // Collapsed-coordinate product rule on T.
  //   a-direction : Gauss-Legendre            (weight 1)
  //   b-direction : Gauss-Jacobi with a=1,b=0 (absorbs the (1-b) Jacobian)
  //   (r,s) = ( (1+a)(1-b)/2 - 1 , b ),  dr ds = (1-b)/2 da db
  //
  // n1d = p + 1 is the minimum for an exact mass matrix (degree 2p).
  const int n1d = p + 1 + overInt_;
  const jacobi::Rule ga = jacobi::gaussJacobi(n1d, 0.0, 0.0);
  const jacobi::Rule gb = jacobi::gaussJacobi(n1d, 1.0, 0.0);

  nq_ = n1d * n1d;
  rq_.resize(nq_);
  sq_.resize(nq_);
  wq_.resize(nq_);

  int q = 0;
  for (int ia = 0; ia < n1d; ++ia) {
    for (int ib = 0; ib < n1d; ++ib, ++q) {
      const Real a = ga.x[ia], b = gb.x[ib];
      rq_[q] = 0.5 * (1.0 + a) * (1.0 - b) - 1.0;
      sq_[q] = b;
      wq_[q] = 0.5 * ga.w[ia] * gb.w[ib];
    }
  }

  const auto& md = dubiner::modes(p_);
  V_.resize(nq_, nm_);
  Vr_.resize(nq_, nm_);
  Vs_.resize(nq_, nm_);
  for (int k = 0; k < nm_; ++k) {
    for (int qq = 0; qq < nq_; ++qq) {
      V_(qq, k) = dubiner::eval(rq_[qq], sq_[qq], md[k].i, md[k].j);
      Real dr, ds;
      dubiner::grad(rq_[qq], sq_[qq], md[k].i, md[k].j, dr, ds);
      Vr_(qq, k) = dr;
      Vs_(qq, k) = ds;
    }
  }

  Wr_ = Vr_.transpose() * wq_.asDiagonal();
  Ws_ = Vs_.transpose() * wq_.asDiagonal();
}

const Mat& RefTriangle::faceV(int face, int numFaceQuad) const {
  if (face < 0 || face >= FaceGeom::kNumFaces)
    throw std::invalid_argument("faceV: face must be 0..2");
  if (numFaceQuad < 1) throw std::invalid_argument("faceV: numFaceQuad < 1");

  const auto key = std::make_pair(face, numFaceQuad);
  auto it = faceV_.find(key);
  if (it != faceV_.end()) return it->second;

  const jacobi::Rule& g = jacobi::gaussLegendreCached(numFaceQuad);
  const auto& md = dubiner::modes(p_);

  Mat M(numFaceQuad, nm_);
  for (int qq = 0; qq < numFaceQuad; ++qq) {
    Real r, s;
    FaceGeom::param(face, g.x[qq], r, s);
    for (int k = 0; k < nm_; ++k) M(qq, k) = dubiner::eval(r, s, md[k].i, md[k].j);
  }
  return faceV_.emplace(key, std::move(M)).first->second;
}

const Mat& RefTriangle::faceLift(int face, int numFaceQuad) const {
  const auto key = std::make_pair(face, numFaceQuad);
  auto it = faceLift_.find(key);
  if (it != faceLift_.end()) return it->second;

  const jacobi::Rule& g = jacobi::gaussLegendreCached(numFaceQuad);
  Mat L = faceV(face, numFaceQuad).transpose() * g.w.asDiagonal();
  return faceLift_.emplace(key, std::move(L)).first->second;
}

const RefTriangle& refTri(int p, int overIntegration) {
  static std::map<std::pair<int, int>, RefTriangle> cache;
  const auto key = std::make_pair(p, overIntegration);
  auto it = cache.find(key);
  if (it == cache.end()) {
    RefTriangle rt(p, overIntegration);  // private ctor: refTri is a friend
    it = cache.emplace(key, std::move(rt)).first;
  }
  return it->second;
}

}  // namespace dgw
