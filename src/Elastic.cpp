#include "dgwave/Elastic.hpp"

#include <algorithm>
#include <random>
#include <stdexcept>

namespace dgw {

// ---------------------------------------------------------------------------
// ElasticField
// ---------------------------------------------------------------------------

void ElasticField::resize(const Mesh& m) {
  for (auto& comp : d_) {
    comp.assign(static_cast<std::size_t>(m.numElements()), Vec());
    for (int e = 0; e < m.numElements(); ++e)
      comp[static_cast<std::size_t>(e)] = Vec::Zero(dubiner::numModes(m.degree(e)));
  }
}

void ElasticField::setZero() {
  for (auto& comp : d_)
    for (auto& v : comp) v.setZero();
}

void ElasticField::randomize(unsigned seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<Real> u(-1.0, 1.0);
  for (auto& comp : d_)
    for (auto& v : comp)
      for (int i = 0; i < v.size(); ++i) v[i] = u(gen);
}

void ElasticField::axpy(Real a, const ElasticField& x) {
  for (int c = 0; c < kNumComponents; ++c)
    for (std::size_t e = 0; e < d_[c].size(); ++e) d_[c][e] += a * x.d_[c][e];
}

Real ElasticField::dot(const ElasticField& o) const {
  Real acc = 0.0;
  for (int c = 0; c < kNumComponents; ++c)
    for (std::size_t e = 0; e < d_[c].size(); ++e) acc += d_[c][e].dot(o.d_[c][e]);
  return acc;
}

// ---------------------------------------------------------------------------
// Riemann solver
// ---------------------------------------------------------------------------

namespace {

/// Scalar two-field Riemann problem shared by the P and S systems:
///     dv/dt - (1/rho) ds/dn = 0,   ds/dt - M dv/dn = 0,   Z = rho*c.
/// Interface state, exact for arbitrary impedance contrast:
///     v* = (ZL vL + ZR vR + sR - sL) / (ZL + ZR)
///     s* = (ZR sL + ZL sR + ZL ZR (vR - vL)) / (ZL + ZR)
inline void riemann1d(Real vL, Real sL, Real ZL, Real vR, Real sR, Real ZR,
                      Real& vStar, Real& sStar) {
  const Real Zsum = ZL + ZR;
  if (Zsum < 1e-30) {
    // Both sides are shear-free (fluid-fluid). No shear traction can be
    // transmitted; the tangential velocity is decoupled and its value never
    // reaches the flux.
    vStar = 0.5 * (vL + vR);
    sStar = 0.0;
    return;
  }
  vStar = (ZL * vL + ZR * vR + sR - sL) / Zsum;
  sStar = (ZR * sL + ZL * sR + ZL * ZR * (vR - vL)) / Zsum;
}

}  // namespace

RiemannState riemannElastic(Real vnL, Real vtL, Real snnL, Real sntL,
                            Real vnR, Real vtR, Real snnR, Real sntR,
                            const Material& mL, const Material& mR) {
  // A medium with mu = 0 supports no shear traction. That is a constitutive
  // identity, not a numerical safeguard: sigma_nt is not a dynamical variable
  // there (its evolution equation reduces to dsigma_xy/dt = 0), so the storage
  // slot must be read as zero however it was filled. Imposing it here keeps a
  // polluted water-layer state from injecting shear into the solid across a
  // fluid-solid contact.
  if (mL.mu <= 0.0) sntL = 0.0;
  if (mR.mu <= 0.0) sntR = 0.0;

  RiemannState st;
  riemann1d(vnL, snnL, mL.zp(), vnR, snnR, mR.zp(), st.vn, st.snn);
  riemann1d(vtL, sntL, mL.zs(), vtR, sntR, mR.zs(), st.vt, st.snt);
  return st;
}

// ---------------------------------------------------------------------------
// ElasticOperator
// ---------------------------------------------------------------------------

ElasticOperator::ElasticOperator(const Mesh& mesh, MaterialModel mats, Flux flux)
    : mesh_(mesh), mat_(std::move(mats)), flux_(flux) {
  if (static_cast<int>(mat_.size()) != mesh_.numElements())
    throw std::invalid_argument("ElasticOperator: material count != element count");
  bc_.fill(BC::FreeSurface);
}

void ElasticOperator::apply(const ElasticField& q, ElasticField& dq) const {
  dq.resize(mesh_);

  // ----- volume term -------------------------------------------------------
  //   int_K grad(psi) . F dV, with F evaluated at the volume quadrature points.
  //   The element Jacobian cancels between the integral and the mass matrix,
  //   so no J appears here at all.
  for (int e = 0; e < mesh_.numElements(); ++e) {
    const RefTriangle& ref = refTri(mesh_.degree(e));
    const Material& m = mat_[static_cast<std::size_t>(e)];
    const ElementGeom& g = mesh_.geom(e);
    const Real M = m.lambda + 2.0 * m.mu;
    const Real invRho = 1.0 / m.rho;

    std::array<Vec, kNumComponents> qq;
    for (int c = 0; c < kNumComponents; ++c) qq[c] = ref.V() * q(c, e);

    // Fx = A q, Fy = B q
    std::array<Vec, kNumComponents> Fx, Fy;
    Fx[VX]  = -invRho * qq[SXX];
    Fx[VY]  = -invRho * qq[SXY];
    Fx[SXX] = -M * qq[VX];
    Fx[SYY] = -m.lambda * qq[VX];
    Fx[SXY] = -m.mu * qq[VY];

    Fy[VX]  = -invRho * qq[SXY];
    Fy[VY]  = -invRho * qq[SYY];
    Fy[SXX] = -m.lambda * qq[VY];
    Fy[SYY] = -M * qq[VY];
    Fy[SXY] = -m.mu * qq[VX];

    for (int c = 0; c < kNumComponents; ++c)
      dq(c, e) = ref.Wr() * (g.drdx * Fx[c] + g.drdy * Fy[c]) +
                 ref.Ws() * (g.dsdx * Fx[c] + g.dsdy * Fy[c]);
  }

  // ----- surface term ------------------------------------------------------
  for (int fi = 0; fi < mesh_.numFaces(); ++fi) {
    const Face& fc = mesh_.face(fi);
    const int nfq = mesh_.faceQuadSize(fi);

    const int eL = fc.elem[0], lfL = fc.face[0];
    const int eR = fc.elem[1], lfR = fc.face[1];
    const bool interior = !fc.isBoundary();

    const Mat& VfL = refTri(mesh_.degree(eL)).faceV(lfL, nfq);
    std::array<Vec, kNumComponents> qL, qR;
    for (int c = 0; c < kNumComponents; ++c) qL[c] = VfL * q(c, eL);

    if (interior) {
      const Mat& VfR = refTri(mesh_.degree(eR)).faceV(lfR, nfq);
      for (int c = 0; c < kNumComponents; ++c) {
        const Vec t = VfR * q(c, eR);
        qR[c].resize(nfq);
        // Neighbours traverse the shared face in opposite directions: an exact
        // index flip, never an interpolation.
        for (int k = 0; k < nfq; ++k) qR[c][k] = t[FaceGeom::mirrorIndex(nfq, k)];
      }
    }

    const Vec2 n = mesh_.geom(eL).faceNormal[lfL];
    const Real nx = n[0], ny = n[1];
    const Vec2 t2(-ny, nx);  // tangent, right-handed with n

    const Material& mL = mat_[static_cast<std::size_t>(eL)];
    const Material& mR = interior ? mat_[static_cast<std::size_t>(eR)] : mL;
    const BC bc = interior ? BC::FreeSurface : bc_[static_cast<std::size_t>(fc.tag)];

    // Interface velocity and traction at each face quadrature point.
    std::vector<Real> vxS(nfq), vyS(nfq), TxS(nfq), TyS(nfq);

    for (int k = 0; k < nfq; ++k) {
      // Rotate the left trace into the (n,t) frame.
      const Real vnL = qL[VX][k] * nx + qL[VY][k] * ny;
      const Real vtL = qL[VX][k] * t2[0] + qL[VY][k] * t2[1];
      const Real TxL = qL[SXX][k] * nx + qL[SXY][k] * ny;
      const Real TyL = qL[SXY][k] * nx + qL[SYY][k] * ny;
      const Real snnL = TxL * nx + TyL * ny;
      const Real sntL = TxL * t2[0] + TyL * t2[1];

      Real vnR, vtR, snnR, sntR;
      if (interior) {
        vnR = qR[VX][k] * nx + qR[VY][k] * ny;
        vtR = qR[VX][k] * t2[0] + qR[VY][k] * t2[1];
        const Real TxR = qR[SXX][k] * nx + qR[SXY][k] * ny;
        const Real TyR = qR[SXY][k] * nx + qR[SYY][k] * ny;
        snnR = TxR * nx + TyR * ny;
        sntR = TxR * t2[0] + TyR * t2[1];
      } else {
        // Every boundary condition is a ghost state, so one Riemann solver
        // serves all of them and no boundary needs its own code path.
        switch (bc) {
          case BC::FreeSurface:  // T* = 0
            vnR = vnL;  vtR = vtL;  snnR = -snnL; sntR = -sntL; break;
          case BC::Rigid:        // v* = 0
            vnR = -vnL; vtR = -vtL; snnR = snnL;  sntR = sntL;  break;
          case BC::Absorbing:    // zero incoming characteristic
          default:
            vnR = 0.0;  vtR = 0.0;  snnR = 0.0;   sntR = 0.0;   break;
        }
      }

      Real vnS, vtS, snnStar, sntStar;
      if (flux_ == Flux::Centered) {
        vnS = 0.5 * (vnL + vnR);   vtS = 0.5 * (vtL + vtR);
        snnStar = 0.5 * (snnL + snnR);  sntStar = 0.5 * (sntL + sntR);
      } else {
        const RiemannState st =
            riemannElastic(vnL, vtL, snnL, sntL, vnR, vtR, snnR, sntR, mL, mR);
        vnS = st.vn; vtS = st.vt; snnStar = st.snn; sntStar = st.snt;
      }

      // Rotate the interface state back to Cartesian.
      vxS[k] = vnS * nx + vtS * t2[0];
      vyS[k] = vnS * ny + vtS * t2[1];
      TxS[k] = snnStar * nx + sntStar * t2[0];
      TyS[k] = snnStar * ny + sntStar * t2[1];
    }

    // Assemble  -fscale * faceLift * (An_K q*)  into each adjacent element.
    // Note An_K uses that element's OWN material with the SHARED interface
    // state: across a material jump the two sides legitimately see different
    // fluxes, while velocity and traction stay single-valued as the physics
    // requires.
    auto scatter = [&](int e, int lf, bool flip) {
      const Material& m = mat_[static_cast<std::size_t>(e)];
      const ElementGeom& g = mesh_.geom(e);
      const Real M = m.lambda + 2.0 * m.mu;
      const Real sgn = flip ? -1.0 : 1.0;  // the neighbour's outward normal
      const Real enx = sgn * nx, eny = sgn * ny;

      std::array<Vec, kNumComponents> fl;
      for (int c = 0; c < kNumComponents; ++c) fl[c].resize(nfq);

      for (int k = 0; k < nfq; ++k) {
        // Index k is measured along element eL's traversal of the face; the
        // neighbour's lift expects its own ordering.
        const int j = flip ? FaceGeom::mirrorIndex(nfq, k) : k;
        const Real vx = vxS[j], vy = vyS[j], Tx = sgn * TxS[j], Ty = sgn * TyS[j];
        fl[VX][k]  = -Tx / m.rho;
        fl[VY][k]  = -Ty / m.rho;
        fl[SXX][k] = -(M * vx * enx + m.lambda * vy * eny);
        fl[SYY][k] = -(m.lambda * vx * enx + M * vy * eny);
        fl[SXY][k] = -m.mu * (vy * enx + vx * eny);
      }
      const Mat& L = refTri(mesh_.degree(e)).faceLift(lf, nfq);
      for (int c = 0; c < kNumComponents; ++c)
        dq(c, e) -= g.fscale[lf] * (L * fl[c]);
    };

    scatter(eL, lfL, false);
    if (interior) scatter(eR, lfR, true);
  }
}

Real ElasticOperator::maxTimeStep(Real cfl) const {
  Real dt = std::numeric_limits<Real>::max();
  for (int e = 0; e < mesh_.numElements(); ++e) {
    const int p = mesh_.degree(e);
    const Real v = mat_[static_cast<std::size_t>(e)].vp();
    dt = std::min(dt, mesh_.geom(e).inradius / (v * (2.0 * p + 1.0)));
  }
  return cfl * dt;
}

namespace {

/// Strain from stress (plane strain compliance), applied to modal coefficients.
/// Legitimate because the material is constant over the element, so the map is
/// a constant matrix acting componentwise on the expansion.
inline void stressToStrain(const Material& m, const Vec& sxx, const Vec& syy,
                           const Vec& sxy, Vec& exx, Vec& eyy, Vec& exy) {
  if (m.mu <= 0.0) {
    // In a fluid the compliance is singular: deviatoric stress cannot exist, so
    // deviatoric strain is not determined by stress at all. Only the volumetric
    // branch survives, and it gives the correct strain energy p^2/(2*lambda) on
    // the admissible set sigma_xx = sigma_yy, sigma_xy = 0.
    const Vec vol = (sxx + syy) / (4.0 * m.lambda);
    exx = vol;
    eyy = vol;
    exy = Vec::Zero(sxy.size());
    return;
  }
  const Real M = m.lambda + 2.0 * m.mu;
  const Real den = 4.0 * m.mu * (m.lambda + m.mu);
  exx = (M * sxx - m.lambda * syy) / den;
  eyy = (M * syy - m.lambda * sxx) / den;
  exy = sxy / (2.0 * m.mu);
}

}  // namespace

void ElasticOperator::enforceFluidConstraint(ElasticField& q) const {
  for (int e = 0; e < mesh_.numElements(); ++e) {
    if (mat_[static_cast<std::size_t>(e)].mu > 0.0) continue;
    const Vec mean = 0.5 * (q(SXX, e) + q(SYY, e));
    q(SXX, e) = mean;
    q(SYY, e) = mean;
    q(SXY, e).setZero();
  }
}

Real ElasticOperator::energy(const ElasticField& q) const {
  Real E = 0.0;
  for (int e = 0; e < mesh_.numElements(); ++e) {
    const Material& m = mat_[static_cast<std::size_t>(e)];
    const Real J = mesh_.geom(e).J;
    Vec exx, eyy, exy;
    stressToStrain(m, q(SXX, e), q(SYY, e), q(SXY, e), exx, eyy, exy);
    const Real kin = m.rho * (q(VX, e).squaredNorm() + q(VY, e).squaredNorm());
    const Real pot = q(SXX, e).dot(exx) + q(SYY, e).dot(eyy) +
                     2.0 * q(SXY, e).dot(exy);
    E += 0.5 * J * (kin + pot);
  }
  return E;
}

Real ElasticOperator::energyRate(const ElasticField& q) const {
  ElasticField dq;
  apply(q, dq);
  Real rate = 0.0;
  for (int e = 0; e < mesh_.numElements(); ++e) {
    const Material& m = mat_[static_cast<std::size_t>(e)];
    const Real J = mesh_.geom(e).J;
    Vec exx, eyy, exy;
    stressToStrain(m, q(SXX, e), q(SYY, e), q(SXY, e), exx, eyy, exy);
    rate += J * (m.rho * (q(VX, e).dot(dq(VX, e)) + q(VY, e).dot(dq(VY, e))) +
                 exx.dot(dq(SXX, e)) + eyy.dot(dq(SYY, e)) +
                 2.0 * exy.dot(dq(SXY, e)));
  }
  return rate;
}

}  // namespace dgw
