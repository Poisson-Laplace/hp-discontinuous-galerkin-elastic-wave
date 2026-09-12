// Verification suite for the reference-element layer.
//
// These are not smoke tests. Each one pins down an identity that the DG
// residual silently depends on; if any of them drifts, the scheme loses
// consistency and you will spend a week blaming the flux instead.

#include <cmath>
#include <cstdio>
#include <string>

#include "dgwave/Dubiner.hpp"
#include "dgwave/Jacobi.hpp"
#include "dgwave/RefTriangle.hpp"

using namespace dgw;

namespace {

int g_failures = 0;

void report(const std::string& name, Real err, Real tol) {
  const bool ok = std::isfinite(err) && err <= tol;
  if (!ok) ++g_failures;
  std::printf("  [%s] %-52s  err = %.3e   (tol %.1e)\n", ok ? "PASS" : "FAIL",
              name.c_str(), err, tol);
}

// ---------------------------------------------------------------------------

void testJacobiOrthonormality() {
  std::puts("\n1. Orthonormality of the 1D Jacobi basis");
  const Real alpha = 1.0, beta = 0.0;
  const int nmax = 12;
  const jacobi::Rule g = jacobi::gaussJacobi(nmax + 4, alpha, beta);

  Real worst = 0.0;
  for (int m = 0; m <= nmax; ++m)
    for (int n = 0; n <= nmax; ++n) {
      Real acc = 0.0;
      for (int q = 0; q < g.size(); ++q)
        acc += g.w[q] * jacobi::P(g.x[q], alpha, beta, m) *
               jacobi::P(g.x[q], alpha, beta, n);
      worst = std::max(worst, std::abs(acc - (m == n ? 1.0 : 0.0)));
    }
  report("<P_m, P_n>_(1,0) = delta_mn, degrees 0..12", worst, 1e-12);
}

void testQuadratureExactness() {
  std::puts("\n2. Gauss-Jacobi exactness up to degree 2n-1");
  Real worst = 0.0;
  for (int n = 1; n <= 10; ++n) {
    const jacobi::Rule g = jacobi::gaussLegendre(n);
    for (int k = 0; k <= 2 * n - 1; ++k) {
      Real num = 0.0;
      for (int q = 0; q < g.size(); ++q) num += g.w[q] * std::pow(g.x[q], k);
      // \int_{-1}^{1} x^k dx = 2/(k+1) for even k, 0 for odd k
      const Real exact = (k % 2 == 0) ? 2.0 / (k + 1.0) : 0.0;
      worst = std::max(worst, std::abs(num - exact));
    }
  }
  report("Gauss-Legendre monomials, n = 1..10", worst, 1e-13);

  // The b-direction rule must reproduce \int (1-b) b^k db exactly.
  worst = 0.0;
  for (int n = 1; n <= 10; ++n) {
    const jacobi::Rule g = jacobi::gaussJacobi(n, 1.0, 0.0);
    for (int k = 0; k <= 2 * n - 1; ++k) {
      Real num = 0.0;
      for (int q = 0; q < g.size(); ++q) num += g.w[q] * std::pow(g.x[q], k);
      const Real mk  = (k % 2 == 0) ? 2.0 / (k + 1.0) : 0.0;         // \int b^k
      const Real mk1 = ((k + 1) % 2 == 0) ? 2.0 / (k + 2.0) : 0.0;   // \int b^{k+1}
      worst = std::max(worst, std::abs(num - (mk - mk1)));
    }
  }
  report("Gauss-Jacobi(1,0) monomials, n = 1..10", worst, 1e-13);
}

void testAreaAndMass() {
  std::puts("\n3. Reference triangle: area and mass matrix");
  Real worstArea = 0.0, worstMass = 0.0;
  for (int p = 0; p <= 10; ++p) {
    const RefTriangle& e = refTri(p);
    worstArea = std::max(worstArea, std::abs(e.wq().sum() - 2.0));

    // M = V^T diag(w) V must be the identity: this is the single property the
    // whole "never invert a mass matrix" design rests on.
    const Mat M = e.V().transpose() * e.wq().asDiagonal() * e.V();
    const Mat I = Mat::Identity(e.numModes(), e.numModes());
    worstMass = std::max(worstMass, (M - I).cwiseAbs().maxCoeff());
  }
  report("sum of volume weights = |T| = 2, p = 0..10", worstArea, 1e-13);
  report("V^T W V = I  (orthonormal mass matrix), p = 0..10", worstMass, 1e-11);
}

void testGradientsAgainstFD() {
  std::puts("\n4. Analytic gradients vs central finite differences");
  const Real h = 1e-5;
  const Real pts[][2] = {{-0.3, -0.4}, {0.1, -0.6}, {-0.7, 0.2},
                         {-0.2, 0.05}, {0.4, -0.8}, {-0.9, -0.9}};
  Real worst = 0.0;
  for (int p = 1; p <= 8; ++p) {
    const auto& md = dubiner::modes(p);
    for (const auto& pt : pts) {
      for (std::size_t k = 0; k < md.size(); ++k) {
        Real dr, ds;
        dubiner::grad(pt[0], pt[1], md[k].i, md[k].j, dr, ds);
        const Real fdr = (dubiner::eval(pt[0] + h, pt[1], md[k].i, md[k].j) -
                          dubiner::eval(pt[0] - h, pt[1], md[k].i, md[k].j)) /
                         (2 * h);
        const Real fds = (dubiner::eval(pt[0], pt[1] + h, md[k].i, md[k].j) -
                          dubiner::eval(pt[0], pt[1] - h, md[k].i, md[k].j)) /
                         (2 * h);
        const Real scale = std::max(1.0, std::max(std::abs(dr), std::abs(ds)));
        worst = std::max(worst, std::max(std::abs(dr - fdr), std::abs(ds - fds)) / scale);
      }
    }
  }
  report("d(psi)/dr, d(psi)/ds vs central FD, p = 1..8", worst, 1e-6);
}

void testDivergenceTheorem() {
  std::puts("\n5. Divergence theorem on the reference element");
  // \int_T grad(psi_i) dT = \oint_{dT} psi_i n ds
  //
  // This is the strongest single check available here: it couples the volume
  // gradient operator, the face quadrature, the face parametrisation, the
  // outward normals and the reference surface Jacobians. If the DG surface
  // and volume terms are ever inconsistent, this is what catches it.
  Real worst = 0.0;
  for (int p = 0; p <= 8; ++p) {
    const RefTriangle& e = refTri(p);
    const int nfq = p + 2;

    const Vec lhsR = e.Vr().transpose() * e.wq();
    const Vec lhsS = e.Vs().transpose() * e.wq();

    Vec rhsR = Vec::Zero(e.numModes());
    Vec rhsS = Vec::Zero(e.numModes());
    for (int f = 0; f < FaceGeom::kNumFaces; ++f) {
      const Vec2 n = FaceGeom::refNormal(f);
      const Real sJ = FaceGeom::refSurfJac(f);
      // faceLift integrates in the face parameter; sJ converts to arc length.
      const Vec int_psi = e.faceLift(f, nfq) * Vec::Ones(nfq);
      rhsR += n[0] * sJ * int_psi;
      rhsS += n[1] * sJ * int_psi;
    }
    worst = std::max(worst, (lhsR - rhsR).cwiseAbs().maxCoeff());
    worst = std::max(worst, (lhsS - rhsS).cwiseAbs().maxCoeff());
  }
  report("int grad(psi) = oint psi*n, p = 0..8", worst, 1e-11);
}

void testHierarchy() {
  std::puts("\n6. Hierarchical mode ordering (p-refinement = zero padding)");
  int mismatch = 0;
  for (int p = 0; p < 10; ++p) {
    const auto& lo = dubiner::modes(p);
    const auto& hi = dubiner::modes(p + 1);
    if (static_cast<int>(lo.size()) != dubiner::numModes(p)) ++mismatch;
    for (std::size_t k = 0; k < lo.size(); ++k)
      if (lo[k].i != hi[k].i || lo[k].j != hi[k].j) ++mismatch;
  }
  report("modes(p) is a prefix of modes(p+1), p = 0..9",
         static_cast<Real>(mismatch), 0.0);

  // The degree-p Vandermonde must be the leading block of the degree-(p+1) one
  // when both are sampled at the same points, which is what makes projection
  // between neighbours of different p a no-op.
  const int p = 5;
  const RefTriangle& hi = refTri(p + 1);
  const auto& mdLo = dubiner::modes(p);
  Real worst = 0.0;
  for (int q = 0; q < hi.numVolQuad(); ++q)
    for (std::size_t k = 0; k < mdLo.size(); ++k) {
      const Real direct = dubiner::eval(hi.rq()[q], hi.sq()[q], mdLo[k].i, mdLo[k].j);
      worst = std::max(worst, std::abs(direct - hi.V()(q, static_cast<int>(k))));
    }
  report("leading block of V(p+1) equals the degree-p basis", worst, 1e-14);
}

void testProjectionAndFaceTrace() {
  std::puts("\n7. L2 projection and face traces");
  // f is degree 3, so for p >= 3 the modal projection is exact and the face
  // trace must reproduce f along each edge to machine precision.
  auto f = [](Real r, Real s) { return r * r * s + 3.0 * r - s * s + 0.5; };

  const int p = 4;
  const RefTriangle& e = refTri(p);

  Vec fq(e.numVolQuad());
  for (int q = 0; q < e.numVolQuad(); ++q) fq[q] = f(e.rq()[q], e.sq()[q]);

  // Mass matrix is the identity, so the projection is one weighted transpose.
  const Vec uhat = e.V().transpose() * (e.wq().asDiagonal() * fq);

  Real volErr = (e.V() * uhat - fq).cwiseAbs().maxCoeff();
  report("volume reconstruction of a degree-3 polynomial", volErr, 1e-12);

  const int nfq = 6;
  const jacobi::Rule& g = jacobi::gaussLegendreCached(nfq);
  Real faceErr = 0.0;
  for (int fc = 0; fc < FaceGeom::kNumFaces; ++fc) {
    const Vec trace = e.faceV(fc, nfq) * uhat;
    for (int q = 0; q < nfq; ++q) {
      Real r, s;
      FaceGeom::param(fc, g.x[q], r, s);
      faceErr = std::max(faceErr, std::abs(trace[q] - f(r, s)));
    }
  }
  report("face traces match the analytic function on all 3 edges", faceErr, 1e-12);

  // Neighbour index flip must land on the same physical point.
  Real mirrorErr = 0.0;
  for (int q = 0; q < nfq; ++q)
    mirrorErr = std::max(mirrorErr,
                         std::abs(g.x[q] + g.x[FaceGeom::mirrorIndex(nfq, q)]));
  report("mirrorIndex maps a face node to its neighbour's node", mirrorErr, 1e-14);
}

}  // namespace

int main() {
  std::puts("=====================================================================");
  std::puts(" dgwave - reference element verification");
  std::puts("=====================================================================");

  testJacobiOrthonormality();
  testQuadratureExactness();
  testAreaAndMass();
  testGradientsAgainstFD();
  testDivergenceTheorem();
  testHierarchy();
  testProjectionAndFaceTrace();

  std::puts("\n---------------------------------------------------------------------");
  if (g_failures == 0) {
    std::puts(" ALL CHECKS PASSED");
    return 0;
  }
  std::printf(" %d CHECK(S) FAILED\n", g_failures);
  return 1;
}
