// Verification suite for the mesh layer.
//
// The load-bearing check here is test 5: the two sides of a shared face must
// agree on WHICH physical point each quadrature index refers to. Every numerical
// flux in the engine assumes it. If that mapping is off, the scheme still runs,
// still looks stable, and quietly couples the wrong points.

#include <cmath>
#include <cstdio>
#include <set>
#include <string>

#include "dgwave/Jacobi.hpp"
#include "dgwave/Mesh.hpp"
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

Mesh makeMesh(int p) { return Mesh::rectangle(0.0, 1500.0, 0.0, 900.0, 12, 8, p); }

// ---------------------------------------------------------------------------

void testOrientationAndArea() {
  std::puts("\n1. Orientation, area and metric inversion");
  const Mesh m = makeMesh(3);

  Real worstJ = 0.0;
  for (int e = 0; e < m.numElements(); ++e)
    worstJ = std::min(worstJ, m.geom(e).J);
  report("every Jacobian is positive (min J shown negated)", -worstJ, 0.0);

  report("sum of 2*J equals the domain area",
         std::abs(m.area() - 1500.0 * 900.0) / (1500.0 * 900.0), 1e-13);

  // The metric terms must invert the map exactly:
  //   [dr/dx dr/dy; ds/dx ds/dy] * [dx/dr dx/ds; dy/dr dy/ds] = I
  Real worst = 0.0;
  for (int e = 0; e < m.numElements(); ++e) {
    const auto& t = m.element(e);
    const Vec2 A = m.vertices()[t[0]], B = m.vertices()[t[1]], C = m.vertices()[t[2]];
    const Real dxdr = 0.5 * (B[0] - A[0]), dxds = 0.5 * (C[0] - A[0]);
    const Real dydr = 0.5 * (B[1] - A[1]), dyds = 0.5 * (C[1] - A[1]);
    const auto& g = m.geom(e);
    worst = std::max(worst, std::abs(g.drdx * dxdr + g.drdy * dydr - 1.0));
    worst = std::max(worst, std::abs(g.drdx * dxds + g.drdy * dyds - 0.0));
    worst = std::max(worst, std::abs(g.dsdx * dxdr + g.dsdy * dydr - 0.0));
    worst = std::max(worst, std::abs(g.dsdx * dxds + g.dsdy * dyds - 1.0));
  }
  report("metric terms invert the affine map", worst, 1e-13);
}

void testConnectivity() {
  std::puts("\n2. Face connectivity");
  const int nx = 12, ny = 8;
  const Mesh m = makeMesh(3);

  // Euler for a triangulated disk: E = V + F - 1.
  const int expectedFaces = (nx + 1) * (ny + 1) + 2 * nx * ny - 1;
  report("face count matches Euler's formula",
         std::abs(static_cast<Real>(m.numFaces() - expectedFaces)), 0.0);

  int nBoundary = 0, taggedWrong = 0;
  for (const auto& f : m.faces()) {
    if (f.isBoundary()) {
      ++nBoundary;
      if (f.tag == kInterior) ++taggedWrong;
    } else if (f.tag != kInterior) {
      ++taggedWrong;
    }
  }
  report("boundary face count = 2(nx+ny)",
         std::abs(static_cast<Real>(nBoundary - 2 * (nx + ny))), 0.0);
  report("boundary faces tagged, interior faces untagged",
         static_cast<Real>(taggedWrong), 0.0);

  // Every element must reference three distinct faces, and the back-reference
  // from the face must land on that element again.
  int bad = 0;
  for (int e = 0; e < m.numElements(); ++e) {
    std::set<int> ids;
    for (int lf = 0; lf < 3; ++lf) {
      const int gid = m.elemFace(e, lf);
      ids.insert(gid);
      const Face& f = m.face(gid);
      const bool side0 = (f.elem[0] == e && f.face[0] == lf);
      const bool side1 = (f.elem[1] == e && f.face[1] == lf);
      if (!side0 && !side1) ++bad;
    }
    if (ids.size() != 3) ++bad;
  }
  report("element <-> face back-references are consistent",
         static_cast<Real>(bad), 0.0);
}

void testNormals() {
  std::puts("\n3. Normals");
  const Mesh m = makeMesh(3);

  // A closed polygon: the outward normals weighted by edge length must cancel.
  Real worstClosed = 0.0;
  for (int e = 0; e < m.numElements(); ++e) {
    const auto& g = m.geom(e);
    Vec2 acc(0.0, 0.0);
    for (int f = 0; f < 3; ++f) acc += g.faceLength[f] * g.faceNormal[f];
    worstClosed = std::max(worstClosed, acc.norm() / std::sqrt(2.0 * g.J));
  }
  report("sum of L*n over the three faces vanishes", worstClosed, 1e-13);

  // Across a shared face the two sides must see opposite normals and the same
  // edge length.
  Real worstOpp = 0.0, worstLen = 0.0;
  for (const auto& f : m.faces()) {
    if (f.isBoundary()) continue;
    const Vec2 nL = m.geom(f.elem[0]).faceNormal[f.face[0]];
    const Vec2 nR = m.geom(f.elem[1]).faceNormal[f.face[1]];
    const Real LL = m.geom(f.elem[0]).faceLength[f.face[0]];
    const Real LR = m.geom(f.elem[1]).faceLength[f.face[1]];
    worstOpp = std::max(worstOpp, (nL + nR).norm());
    worstLen = std::max(worstLen, std::abs(LL - LR) / LL);
  }
  report("n_left = -n_right on every interior face", worstOpp, 1e-13);
  report("both sides agree on the edge length", worstLen, 1e-14);
}

void testPhysicalDivergence() {
  std::puts("\n4. Divergence theorem on physical elements");
  // \int_K grad(phi_i) dK = \oint_{dK} phi_i n ds, with phi_i(x,y) the modal
  // basis pulled back through the affine map. Dividing by J turns the surface
  // factor into exactly fscale, so this pins down the metric terms, the face
  // normals and fscale in one identity.
  const int p = 4;
  const Mesh m = makeMesh(p);
  const RefTriangle& ref = refTri(p);

  Real worst = 0.0;
  for (int e = 0; e < m.numElements(); ++e) {
    const auto& g = m.geom(e);
    const Vec volR = ref.Vr().transpose() * ref.wq();
    const Vec volS = ref.Vs().transpose() * ref.wq();
    const Vec lhsX = g.drdx * volR + g.dsdx * volS;
    const Vec lhsY = g.drdy * volR + g.dsdy * volS;

    Vec rhsX = Vec::Zero(ref.numModes());
    Vec rhsY = Vec::Zero(ref.numModes());
    for (int lf = 0; lf < 3; ++lf) {
      const int nfq = m.faceQuadSize(m.elemFace(e, lf));
      const Vec intPsi = ref.faceLift(lf, nfq) * Vec::Ones(nfq);
      rhsX += g.faceNormal[lf][0] * g.fscale[lf] * intPsi;
      rhsY += g.faceNormal[lf][1] * g.fscale[lf] * intPsi;
    }
    worst = std::max(worst, (lhsX - rhsX).cwiseAbs().maxCoeff());
    worst = std::max(worst, (lhsY - rhsY).cwiseAbs().maxCoeff());
  }
  report("int grad(phi) = oint phi*n, all elements", worst, 1e-11);
}

void testFacePointMatching() {
  std::puts("\n5. Shared-face quadrature point matching");
  // The flux couples index q on one side with mirrorIndex(q) on the other.
  // Those two indices must denote the SAME physical point.
  const Mesh m = makeMesh(3);

  Real worst = 0.0, scale = 0.0;

  for (int fi = 0; fi < m.numFaces(); ++fi) {
    const Face& f = m.face(fi);
    if (f.isBoundary()) continue;
    const int nfq = m.faceQuadSize(fi);
    const jacobi::Rule& g = jacobi::gaussLegendreCached(nfq);

    for (int q = 0; q < nfq; ++q) {
      Real r0, s0, r1, s1;
      FaceGeom::param(f.face[0], g.x[q], r0, s0);
      FaceGeom::param(f.face[1], g.x[FaceGeom::mirrorIndex(nfq, q)], r1, s1);
      const Vec2 P0 = m.toPhysical(f.elem[0], r0, s0);
      const Vec2 P1 = m.toPhysical(f.elem[1], r1, s1);
      worst = std::max(worst, (P0 - P1).norm());
      scale = std::max(scale, P0.norm());
    }
  }
  report("mirrored quadrature indices land on the same point", worst / scale, 1e-14);
}

void testPhysicalGradient() {
  std::puts("\n6. Physical gradient of a linear field");
  // f(x,y) = 3x - 2y + 1 projected onto the modal basis. Differentiating with
  // the metric terms must return the exact constants 3 and -2 everywhere.
  const int p = 3;
  const Mesh m = makeMesh(p);
  const RefTriangle& ref = refTri(p);
  auto f = [](Real x, Real y) { return 3.0 * x - 2.0 * y + 1.0; };

  Real worst = 0.0;
  for (int e = 0; e < m.numElements(); ++e) {
    Vec fq(ref.numVolQuad());
    for (int q = 0; q < ref.numVolQuad(); ++q) {
      const Vec2 X = m.toPhysical(e, ref.rq()[q], ref.sq()[q]);
      fq[q] = f(X[0], X[1]);
    }
    const Vec uhat = ref.V().transpose() * (ref.wq().asDiagonal() * fq);

    const auto& g = m.geom(e);
    const Vec dr = ref.Vr() * uhat;
    const Vec ds = ref.Vs() * uhat;
    const Vec dfdx = g.drdx * dr + g.dsdx * ds;
    const Vec dfdy = g.drdy * dr + g.dsdy * ds;

    worst = std::max(worst, (dfdx.array() - 3.0).abs().maxCoeff());
    worst = std::max(worst, (dfdy.array() + 2.0).abs().maxCoeff());
  }
  report("recovered df/dx = 3 and df/dy = -2 exactly", worst, 1e-10);
}

void testHpMixedDegrees() {
  std::puts("\n7. Mixed polynomial degrees (hp)");
  Mesh m = makeMesh(2);
  for (int e = 0; e < m.numElements(); ++e) m.setDegree(e, 2 + (e % 4));

  int bad = 0;
  for (int fi = 0; fi < m.numFaces(); ++fi) {
    const Face& f = m.face(fi);
    int expect = m.degree(f.elem[0]);
    if (!f.isBoundary()) expect = std::max(expect, m.degree(f.elem[1]));
    if (m.faceQuadSize(fi, 1) != expect + 2) ++bad;
  }
  report("face rule sized from max(pL, pR)", static_cast<Real>(bad), 0.0);

  // The point-matching identity must survive neighbours of different degree,
  // since the shared rule is what makes the flux well defined.
  Real worst = 0.0, scale = 0.0;
  for (int fi = 0; fi < m.numFaces(); ++fi) {
    const Face& f = m.face(fi);
    if (f.isBoundary()) continue;
    const int nfq = m.faceQuadSize(fi);
    const jacobi::Rule& g = jacobi::gaussLegendreCached(nfq);
    for (int q = 0; q < nfq; ++q) {
      Real r0, s0, r1, s1;
      FaceGeom::param(f.face[0], g.x[q], r0, s0);
      FaceGeom::param(f.face[1], g.x[FaceGeom::mirrorIndex(nfq, q)], r1, s1);
      const Vec2 P0 = m.toPhysical(f.elem[0], r0, s0);
      const Vec2 P1 = m.toPhysical(f.elem[1], r1, s1);
      worst = std::max(worst, (P0 - P1).norm());
      scale = std::max(scale, P0.norm());
    }
    // Both degrees must be able to evaluate on that shared rule.
    refTri(m.degree(f.elem[0])).faceV(f.face[0], nfq);
    refTri(m.degree(f.elem[1])).faceV(f.face[1], nfq);
  }
  report("point matching holds across differing degrees", worst / scale, 1e-14);
}

}  // namespace

int main() {
  std::puts("=====================================================================");
  std::puts(" dgwave - mesh verification");
  std::puts("=====================================================================");

  testOrientationAndArea();
  testConnectivity();
  testNormals();
  testPhysicalDivergence();
  testFacePointMatching();
  testPhysicalGradient();
  testHpMixedDegrees();

  std::puts("\n---------------------------------------------------------------------");
  if (g_failures == 0) {
    std::puts(" ALL CHECKS PASSED");
    return 0;
  }
  std::printf(" %d CHECK(S) FAILED\n", g_failures);
  return 1;
}
