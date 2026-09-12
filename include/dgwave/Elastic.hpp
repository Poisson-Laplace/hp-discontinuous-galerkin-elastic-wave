#pragma once
#include <array>
#include <cmath>
#include <vector>

#include "dgwave/Mesh.hpp"
#include "dgwave/Output.hpp"
#include "dgwave/Types.hpp"

namespace dgw {

// 2D isotropic elastodynamics, velocity-stress form, plane strain:
//
//   rho dv_x/dt = d(sxx)/dx + d(sxy)/dy
//   rho dv_y/dt = d(sxy)/dx + d(syy)/dy
//       dsxx/dt = (l+2m) dv_x/dx +     l dv_y/dy
//       dsyy/dt =     l dv_x/dx + (l+2m) dv_y/dy
//       dsxy/dt =     m (dv_x/dy + dv_y/dx)
//
// written as  dq/dt + A dq/dx + B dq/dy = 0  with q = (vx, vy, sxx, syy, sxy)
// and flux functions Fx = A q, Fy = B q. Tension is positive.

enum Component : int { VX = 0, VY = 1, SXX = 2, SYY = 3, SXY = 4 };
inline constexpr int kNumComponents = 5;

/// Element-wise constant isotropic material. Constant per element is not a
/// simplification of convenience: it is the granularity the FWI gradient is
/// taken at, so the forward and inverse problems share one parameterisation.
struct Material {
  Real rho = 1.0;
  Real lambda = 1.0;
  Real mu = 1.0;

  Real vp() const { return std::sqrt((lambda + 2.0 * mu) / rho); }
  Real vs() const { return std::sqrt(mu / rho); }
  Real zp() const { return rho * vp(); }   ///< P impedance
  Real zs() const { return rho * vs(); }   ///< S impedance; zero in a fluid

  static Material fromVelocities(Real vp, Real vs, Real rho) {
    return Material{rho, rho * (vp * vp - 2.0 * vs * vs), rho * vs * vs};
  }
};

using MaterialModel = std::vector<Material>;

/// The five modal fields of the solution, one coefficient vector per element
/// per component. Mixed degrees are stored without padding.
class ElasticField {
 public:
  ElasticField() = default;
  explicit ElasticField(const Mesh& m) { resize(m); }
  void resize(const Mesh& m);
  void setZero();

  Vec& operator()(int comp, int e) { return d_[comp][static_cast<std::size_t>(e)]; }
  const Vec& operator()(int comp, int e) const {
    return d_[comp][static_cast<std::size_t>(e)];
  }

  const io::ModalField& component(int comp) const { return d_[comp]; }
  int numElements() const { return static_cast<int>(d_[0].size()); }

  /// Fill with pseudo-random coefficients. Used by the verification tests,
  /// where a generic state exercises far more of the operator than a smooth one.
  void randomize(unsigned seed);

  void axpy(Real a, const ElasticField& x);   ///< *this += a * x
  Real dot(const ElasticField& o) const;      ///< plain coefficient inner product

 private:
  std::array<io::ModalField, kNumComponents> d_;
};

/// How the interface state is chosen.
enum class Flux {
  Godunov,  ///< exact elastic Riemann solver; upwind, dissipative, production
  Centered  ///< plain average; exactly energy conserving, used to verify the
            ///< assembly independently of the dissipation mechanism
};

/// What a boundary face does to the incoming characteristic.
enum class BC {
  FreeSurface,  ///< traction goes to zero at the interface: T* = 0
  Absorbing,    ///< zero incoming characteristic (first order)
  Rigid         ///< velocity goes to zero at the interface: v* = 0
};

/// Semi-discrete DG right-hand side.
///
/// apply() is a LINEAR operator in q. That is deliberate and load-bearing: the
/// discrete adjoint is then exactly its transpose, so the inverse problem needs
/// no separately written and separately buggy adjoint code path.
class ElasticOperator {
 public:
  ElasticOperator(const Mesh& mesh, MaterialModel mats, Flux flux = Flux::Godunov);

  void setBC(int tag, BC bc) { bc_[tag] = bc; }
  void setFlux(Flux f) { flux_ = f; }

  /// dq/dt = apply(q).
  void apply(const ElasticField& q, ElasticField& dq) const;

  /// Largest stable time step for an explicit RK4 integrator.
  /// dt = cfl * min_e [ inradius_e / (vp_e * (2p+1)) ]; the (2p+1) is the usual
  /// high-order spectral-radius penalty.
  Real maxTimeStep(Real cfl = 0.5) const;

  /// Project a field onto the physically admissible set: in every fluid
  /// element (mu = 0) the deviatoric stress is removed, leaving
  /// sigma_xx = sigma_yy and sigma_xy = 0.
  ///
  /// This is an INVARIANT of the continuous equations, not a correction: with
  /// mu = 0 the sigma_xx and sigma_yy equations share one right-hand side and
  /// the sigma_xy equation reduces to dsigma_xy/dt = 0, so a state that starts
  /// admissible stays admissible. Applying it to initial conditions is what
  /// makes the water layer well posed.
  void enforceFluidConstraint(ElasticField& q) const;

  /// Total energy  E = int [ rho|v|^2/2 + strain energy ] dV.
  /// With element-constant material and an orthonormal basis this is exact
  /// without quadrature: an L2 inner product of coefficients scaled by J.
  Real energy(const ElasticField& q) const;

  /// dE/dt evaluated from the RHS directly, without any time stepping.
  /// The whole assembly is falsifiable through this one number: Centered flux
  /// with FreeSurface or Rigid walls must return zero to round-off, and Godunov
  /// must return a value that is negative and never positive.
  Real energyRate(const ElasticField& q) const;

  const Mesh& mesh() const { return mesh_; }
  const MaterialModel& materials() const { return mat_; }

 private:
  const Mesh& mesh_;
  MaterialModel mat_;
  Flux flux_;
  std::array<BC, 8> bc_{};
};

/// Exact Riemann solver for the interface between two isotropic half-spaces,
/// in the (normal, tangential) frame. Returns the interface velocity and
/// traction, which is all the numerical flux ever needs: the tangential normal
/// stress sigma_tt rides the zero eigenvalue, stays discontinuous, and never
/// enters An*q.
struct RiemannState {
  Real vn, vt, snn, snt;
};
RiemannState riemannElastic(Real vnL, Real vtL, Real snnL, Real sntL,
                            Real vnR, Real vtR, Real snnR, Real sntR,
                            const Material& mL, const Material& mR);

}  // namespace dgw
