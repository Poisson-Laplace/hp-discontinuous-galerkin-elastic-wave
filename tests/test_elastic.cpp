// Verification suite for the elastic operator.
//
// Every check here compares against a number that is known in advance and could
// have come out wrong. Nothing is inspected for plausibility.
//
// The decisive one is test 5. The semi-discrete energy rate couples the volume
// term, the surface term, the metric factors, the normals, fscale, the
// rotations and the flux. A centered flux with energy-conserving walls MUST
// return zero to round-off; a single sign error anywhere makes it nonzero.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>

#include "dgwave/Elastic.hpp"

using namespace dgw;

namespace {

int g_failures = 0;

void report(const std::string& name, Real err, Real tol) {
  const bool ok = std::isfinite(err) && err <= tol;
  if (!ok) ++g_failures;
  std::printf("  [%s] %-54s  err = %.3e  (tol %.1e)\n", ok ? "PASS" : "FAIL",
              name.c_str(), err, tol);
}

const Material kRock = Material::fromVelocities(3000.0, 1730.0, 2400.0);
const Material kSoft = Material::fromVelocities(1800.0, 1000.0, 2000.0);
const Material kWater = Material::fromVelocities(1500.0, 0.0, 1000.0);

/// Flux Jacobian An = A*nx + B*ny for the ordering (vx,vy,sxx,syy,sxy).
Mat fluxJacobian(const Material& m, Real nx, Real ny) {
  const Real M = m.lambda + 2.0 * m.mu, ir = 1.0 / m.rho;
  Mat An = Mat::Zero(5, 5);
  An(VX, SXX) = -ir * nx;  An(VX, SXY) = -ir * ny;
  An(VY, SXY) = -ir * nx;  An(VY, SYY) = -ir * ny;
  An(SXX, VX) = -M * nx;         An(SXX, VY) = -m.lambda * ny;
  An(SYY, VX) = -m.lambda * nx;  An(SYY, VY) = -M * ny;
  An(SXY, VX) = -m.mu * ny;      An(SXY, VY) = -m.mu * nx;
  return An;
}

// ---------------------------------------------------------------------------

void testEigenstructure() {
  std::puts("\n1. Flux Jacobian spectrum");
  // The eigenvalues must be exactly {-vp, -vs, 0, +vs, +vp} in every direction.
  Real worst = 0.0;
  for (int k = 0; k < 8; ++k) {
    const Real th = k * M_PI / 4.0 + 0.17;
    Mat An = fluxJacobian(kRock, std::cos(th), std::sin(th));
    // Balance by the reference impedance before the decomposition: raw entries
    // span 1/rho ~ 1e-4 to lambda+2mu ~ 1e10, and a condition number of 1e14
    // would make this test measure Eigen's conditioning, not the physics.
    // sigma' = sigma/Z is a similarity transform, so the spectrum is unchanged.
    const Real Z = kRock.zp();
    Vec d(5); d << 1.0, 1.0, Z, Z, Z;
    An = d.cwiseInverse().asDiagonal() * An * d.asDiagonal();
    Eigen::EigenSolver<Mat> es(An);
    Vec ev = es.eigenvalues().real();
    std::sort(ev.data(), ev.data() + ev.size());
    const Real want[5] = {-kRock.vp(), -kRock.vs(), 0.0, kRock.vs(), kRock.vp()};
    for (int i = 0; i < 5; ++i)
      worst = std::max(worst, std::abs(ev[i] - want[i]) / kRock.vp());
    worst = std::max(worst, es.eigenvalues().imag().cwiseAbs().maxCoeff() / kRock.vp());
  }
  report("eigenvalues are {-vp,-vs,0,+vs,+vp}, 8 directions", worst, 1e-12);

  // In a fluid the shear speed collapses and the spectrum degenerates.
  Mat Aw = fluxJacobian(kWater, 1.0, 0.0);
  const Real Zw = kWater.zp();
  Vec dw(5); dw << 1.0, 1.0, Zw, Zw, Zw;
  Aw = dw.cwiseInverse().asDiagonal() * Aw * dw.asDiagonal();
  Eigen::EigenSolver<Mat> esw(Aw);
  Vec evw = esw.eigenvalues().real();
  std::sort(evw.data(), evw.data() + evw.size());
  const Real wantW[5] = {-kWater.vp(), 0.0, 0.0, 0.0, kWater.vp()};
  Real wf = 0.0;
  for (int i = 0; i < 5; ++i) wf = std::max(wf, std::abs(evw[i] - wantW[i]));
  report("fluid (vs=0) collapses to {-vp,0,0,0,+vp}", wf / kWater.vp(), 1e-12);
}

void testRiemannPlaneWave() {
  std::puts("\n2. Riemann solver against analytic wave physics");

  // (a) A single right-going P wave in a uniform medium must pass through the
  //     interface untouched. Its signature is snn = -Zp*vn.
  {
    const Real vn = 0.37, snn = -kRock.zp() * vn;
    const RiemannState st = riemannElastic(vn, 0, snn, 0, vn, 0, snn, 0, kRock, kRock);
    const Real e = std::max(std::abs(st.vn - vn) / std::abs(vn),
                            std::abs(st.snn - snn) / std::abs(snn));
    report("uniform medium leaves a travelling P wave unchanged", e, 1e-14);
  }

  // (b) Impedance contrast. A P wave of unit particle velocity hits the
  //     interface. The reflection coefficient carries OPPOSITE signs for the
  //     two variables -- this is the classic convention trap:
  //         particle velocity   R_v = (Z1 - Z2)/(Z1 + Z2)
  //         normal stress       R_s = (Z2 - Z1)/(Z2 + Z1) = -R_v
  //     The interface state must satisfy both simultaneously, which is why
  //     checking only one of them proves nothing.
  {
    const Real Z1 = kRock.zp(), Z2 = kSoft.zp();
    const Real Rv = (Z1 - Z2) / (Z1 + Z2);
    const Real Rs = (Z2 - Z1) / (Z2 + Z1);
    const Real vInc = 1.0, sInc = -Z1 * vInc;
    const RiemannState st = riemannElastic(vInc, 0, sInc, 0, 0, 0, 0, 0,
                                           kRock, kSoft);
    report("P velocity  reflection R_v = (Z1-Z2)/(Z1+Z2)",
           std::abs(st.vn - (1.0 + Rv) * vInc), 1e-14);
    report("P stress    reflection R_s = (Z2-Z1)/(Z2+Z1)",
           std::abs(st.snn - (1.0 + Rs) * sInc) / std::abs(sInc), 1e-14);
    // Independent cross-check: whatever is transmitted must be a single
    // right-going wave in medium 2, so it obeys sigma = -Z2 * v exactly.
    report("transmitted wave obeys sigma* = -Z2 v*",
           std::abs(st.snn + Z2 * st.vn) / std::abs(sInc), 1e-14);
  }

  // (c) Free surface: an incident P wave must be fully reflected, T* = 0 and
  //     the surface velocity doubled.
  {
    const Real vInc = 1.0, sInc = -kRock.zp() * vInc;
    const RiemannState st =
        riemannElastic(vInc, 0, sInc, 0, vInc, 0, -sInc, 0, kRock, kRock);
    report("free surface: traction vanishes", std::abs(st.snn), 1e-12);
    report("free surface: surface velocity is doubled",
           std::abs(st.vn - 2.0 * vInc), 1e-14);
  }

  // (d) Rigid wall: velocity vanishes, traction doubles.
  {
    const Real vInc = 1.0, sInc = -kRock.zp() * vInc;
    const RiemannState st =
        riemannElastic(vInc, 0, sInc, 0, -vInc, 0, sInc, 0, kRock, kRock);
    report("rigid wall: velocity vanishes", std::abs(st.vn), 1e-14);
    report("rigid wall: traction is doubled",
           std::abs(st.snn - 2.0 * sInc) / std::abs(sInc), 1e-14);
  }

  // (e) Fluid-fluid: no shear traction can be transmitted. This is the water
  //     layer, and the place a naive solver divides by zero.
  {
    const RiemannState st =
        riemannElastic(0.3, 0.5, 1e5, 2e5, -0.2, 0.9, 3e5, 4e5, kWater, kWater);
    report("fluid-fluid transmits no shear traction", std::abs(st.snt), 1e-14);
    report("fluid-fluid P branch stays finite",
           std::isfinite(st.vn) && std::isfinite(st.snn) ? 0.0 : 1.0, 0.0);
  }

  // (f) Solid-fluid: still no shear traction, and the P branch uses both
  //     impedances.
  {
    const RiemannState st =
        riemannElastic(0.3, 0.5, 1e5, 2e5, -0.2, 0.9, 3e5, 4e5, kRock, kWater);
    report("solid-fluid transmits no shear traction", std::abs(st.snt), 1e-14);
  }
}

void testFluxConsistency() {
  std::puts("\n3. Riemann solver consistency");
  // Feeding identical states must return that state: a solver failing this is
  // inconsistent and the scheme loses its order.
  Real worst = 0.0;
  std::mt19937 gen(7);
  std::uniform_real_distribution<Real> u(-1.0, 1.0);
  for (int k = 0; k < 200; ++k) {
    const Real vn = u(gen), vt = u(gen), snn = 1e5 * u(gen), snt = 1e5 * u(gen);
    const RiemannState st = riemannElastic(vn, vt, snn, snt, vn, vt, snn, snt,
                                           kRock, kRock);
    worst = std::max(worst, std::abs(st.vn - vn));
    worst = std::max(worst, std::abs(st.vt - vt));
    worst = std::max(worst, std::abs(st.snn - snn) / 1e5);
    worst = std::max(worst, std::abs(st.snt - snt) / 1e5);
  }
  report("identical left and right states are returned unchanged", worst, 1e-14);
}

void testLinearity() {
  std::puts("\n4. Linearity of the RHS operator");
  // The discrete adjoint is the transpose of apply() only if apply() is exactly
  // linear. This check is what licenses that shortcut later.
  Mesh m = Mesh::rectangle(0.0, 900.0, 0.0, 600.0, 5, 4, 3);
  MaterialModel mats(m.numElements(), kRock);
  for (int e = 0; e < m.numElements(); ++e)
    if (m.toPhysical(e, -1.0 / 3, -1.0 / 3)[1] > 300.0)
      mats[static_cast<std::size_t>(e)] = kSoft;
  ElasticOperator op(m, mats);

  ElasticField a(m), b(m), c(m), da, db, dc;
  a.randomize(1);
  b.randomize(2);
  const Real alpha = 2.7, beta = -1.3;
  c = a; c.axpy(0.0, a);
  for (int cc = 0; cc < kNumComponents; ++cc)
    for (int e = 0; e < m.numElements(); ++e)
      c(cc, e) = alpha * a(cc, e) + beta * b(cc, e);

  op.apply(a, da); op.apply(b, db); op.apply(c, dc);

  Real num = 0.0, den = 0.0;
  for (int cc = 0; cc < kNumComponents; ++cc)
    for (int e = 0; e < m.numElements(); ++e) {
      const Vec diff = dc(cc, e) - (alpha * da(cc, e) + beta * db(cc, e));
      num = std::max(num, diff.cwiseAbs().maxCoeff());
      den = std::max(den, dc(cc, e).cwiseAbs().maxCoeff());
    }
  report("L(a*u + b*v) = a*L(u) + b*L(v)", num / den, 1e-13);

  ElasticField z(m), dz;
  op.apply(z, dz);
  Real zmax = 0.0;
  for (int cc = 0; cc < kNumComponents; ++cc)
    for (int e = 0; e < m.numElements(); ++e)
      zmax = std::max(zmax, dz(cc, e).cwiseAbs().maxCoeff());
  report("L(0) = 0", zmax, 0.0);
}

void testEnergy() {
  std::puts("\n5. Semi-discrete energy identity");
  std::puts("   (volume + surface + metrics + normals + flux, all at once)");

  for (int p : {1, 2, 4}) {
    Mesh m = Mesh::rectangle(0.0, 900.0, 0.0, 600.0, 6, 4, p);
    MaterialModel mats(m.numElements(), kRock);
    // A heterogeneous model, because a uniform one hides errors in the
    // impedance weighting entirely.
    for (int e = 0; e < m.numElements(); ++e) {
      const Vec2 c = m.toPhysical(e, -1.0 / 3, -1.0 / 3);
      if (c[1] > 400.0) mats[static_cast<std::size_t>(e)] = kWater;
      else if (c[0] > 500.0) mats[static_cast<std::size_t>(e)] = kSoft;
    }
    ElasticOperator op(m, mats, Flux::Centered);
    for (int t : {kBottom, kRight, kTop, kLeft}) op.setBC(t, BC::FreeSurface);

    ElasticField q(m);
    q.randomize(42u + static_cast<unsigned>(p));
    // Scale the stresses to a physical magnitude so the two energy terms are
    // comparable and cancellation is actually being tested.
    for (int e = 0; e < m.numElements(); ++e)
      for (int c : {SXX, SYY, SXY}) q(c, e) *= 1e7;
    // A random deviatoric stress inside the water layer is not a state the
    // equations can produce, and its strain energy is genuinely infinite.
    op.enforceFluidConstraint(q);

    const Real E = op.energy(q);
    const Real rateC = op.energyRate(q);
    // Non-dimensionalise by E and a transit time so the tolerance is meaningful.
    const Real tScale = 900.0 / kRock.vp();
    report("centered flux + free surface: dE/dt = 0   (p=" + std::to_string(p) + ")",
           std::abs(rateC) * tScale / E, 1e-11);

    for (int t : {kBottom, kRight, kTop, kLeft}) op.setBC(t, BC::Rigid);
    report("centered flux + rigid walls:  dE/dt = 0   (p=" + std::to_string(p) + ")",
           std::abs(op.energyRate(q)) * tScale / E, 1e-11);

    // Godunov must dissipate. The sign is the whole point: an upwind flux with
    // a flipped sign produces energy and blows up only much later.
    op.setFlux(Flux::Godunov);
    for (int t : {kBottom, kRight, kTop, kLeft}) op.setBC(t, BC::FreeSurface);
    const Real rateG = op.energyRate(q);
    report("Godunov flux dissipates: dE/dt < 0        (p=" + std::to_string(p) + ")",
           rateG > 0.0 ? rateG * tScale / E : 0.0, 0.0);

    // absorbing walls, still Godunov
    for (int t : {kBottom, kRight, kTop, kLeft}) op.setBC(t, BC::Absorbing);
    const Real rateA = op.energyRate(q);
    report("absorbing walls dissipate: dE/dt < 0      (p=" + std::to_string(p) + ")",
           rateA > 0.0 ? rateA * tScale / E : 0.0, 0.0);
  }
}

void testHpEnergy() {
  std::puts("\n6. Energy identity with mixed degrees");
  // If the two sides of a face disagreed about the shared quadrature rule, the
  // surface terms would not telescope and this would be nonzero.
  Mesh m = Mesh::rectangle(0.0, 900.0, 0.0, 600.0, 6, 4, 2);
  for (int e = 0; e < m.numElements(); ++e) m.setDegree(e, 1 + (e % 5));
  MaterialModel mats(m.numElements(), kRock);
  for (int e = 0; e < m.numElements(); ++e)
    if (e % 3 == 0) mats[static_cast<std::size_t>(e)] = kSoft;

  ElasticOperator op(m, mats, Flux::Centered);
  ElasticField q(m);
  q.randomize(9);
  for (int e = 0; e < m.numElements(); ++e)
    for (int c : {SXX, SYY, SXY}) q(c, e) *= 1e7;

  const Real tScale = 900.0 / kRock.vp();
  report("hp mesh, centered flux: dE/dt = 0",
         std::abs(op.energyRate(q)) * tScale / op.energy(q), 1e-11);
}

void testTimeStep() {
  std::puts("\n7. CFL estimate");
  Mesh m = Mesh::rectangle(0.0, 1500.0, 0.0, 900.0, 30, 18, 4);
  MaterialModel mats(m.numElements(), kRock);
  ElasticOperator op(m, mats);
  const Real dt = op.maxTimeStep(0.5);
  const Real hmin = m.geom(0).inradius;
  const Real want = 0.5 * hmin / (kRock.vp() * 9.0);
  report("dt = cfl * inradius / (vp * (2p+1))", std::abs(dt - want) / want, 1e-13);
  std::printf("         (h_in = %.2f m, vp = %.0f m/s, p = 4  ->  dt = %.3e s)\n",
              hmin, kRock.vp(), dt);
}

}  // namespace

int main() {
  std::puts("=====================================================================");
  std::puts(" dgwave - elastic operator verification");
  std::puts("=====================================================================");

  testEigenstructure();
  testRiemannPlaneWave();
  testFluxConsistency();
  testLinearity();
  testEnergy();
  testHpEnergy();
  testTimeStep();

  std::puts("\n---------------------------------------------------------------------");
  if (g_failures == 0) {
    std::puts(" ALL CHECKS PASSED");
    return 0;
  }
  std::printf(" %d CHECK(S) FAILED\n", g_failures);
  return 1;
}
