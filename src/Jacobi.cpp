#include "dgwave/Jacobi.hpp"

#include <cmath>
#include <map>
#include <stdexcept>

namespace dgw::jacobi {

namespace {

/// gamma_0 = \int_{-1}^{1} (1-x)^a (1+x)^b dx
Real mu0(Real a, Real b) {
  return std::pow(2.0, a + b + 1.0) / (a + b + 1.0) * std::tgamma(a + 1.0) *
         std::tgamma(b + 1.0) / std::tgamma(a + b + 1.0);
}

}  // namespace

Real P(Real x, Real a, Real b, int n) {
  if (n < 0) throw std::invalid_argument("jacobi::P: negative degree");

  const Real g0 = mu0(a, b);
  const Real p0 = 1.0 / std::sqrt(g0);
  if (n == 0) return p0;

  const Real g1 = (a + 1.0) * (b + 1.0) / (a + b + 3.0) * g0;
  const Real p1 = ((a + b + 2.0) * x * 0.5 + (a - b) * 0.5) / std::sqrt(g1);
  if (n == 1) return p1;

  // Three-term recurrence in orthonormal form:
  //   a_{k-1} P_{k-1} + (b_k - x) P_k + a_k P_{k+1} = 0
  Real aold = 2.0 / (2.0 + a + b) * std::sqrt((a + 1.0) * (b + 1.0) / (a + b + 3.0));
  Real pm = p0, pc = p1, pn = 0.0;
  for (int i = 1; i <= n - 1; ++i) {
    const Real h1 = 2.0 * i + a + b;
    const Real anew = 2.0 / (h1 + 2.0) *
                      std::sqrt((i + 1.0) * (i + 1.0 + a + b) * (i + 1.0 + a) *
                                (i + 1.0 + b) / ((h1 + 1.0) * (h1 + 3.0)));
    const Real bnew = -(a * a - b * b) / (h1 * (h1 + 2.0));
    pn = (-aold * pm + (x - bnew) * pc) / anew;
    aold = anew;
    pm = pc;
    pc = pn;
  }
  return pc;
}

Real dP(Real x, Real a, Real b, int n) {
  if (n == 0) return 0.0;
  return std::sqrt(n * (n + a + b + 1.0)) * P(x, a + 1.0, b + 1.0, n - 1);
}

Rule gaussJacobi(int n, Real a, Real b) {
  if (n < 1) throw std::invalid_argument("gaussJacobi: n must be >= 1");

  // Symmetric tridiagonal Jacobi matrix of the orthonormal recurrence.
  Vec diag(n);
  Vec sub(n > 1 ? n - 1 : 0);

  for (int i = 0; i < n; ++i) {
    const Real h1 = 2.0 * i + a + b;
    // Diagonal of the symmetric Jacobi matrix:
    //     alpha_k = (b^2 - a^2) / [ (2k+a+b)(2k+a+b+2) ]
    // h1 == 0 can only happen at i == 0 with a + b == 0; there the removable
    // singularity has the limit (b - a)/(a + b + 2).
    diag[i] = (std::abs(h1) < 1e-14)
                  ? (b - a) / (a + b + 2.0)
                  : (b * b - a * a) / (h1 * (h1 + 2.0));
  }
  for (int i = 0; i < n - 1; ++i) {
    const Real h1 = 2.0 * i + a + b;
    const Real j = i + 1.0;
    sub[i] = 2.0 / (h1 + 2.0) *
             std::sqrt(j * (j + a + b) * (j + a) * (j + b) /
                       ((h1 + 1.0) * (h1 + 3.0)));
  }

  Eigen::SelfAdjointEigenSolver<Mat> es;
  es.computeFromTridiagonal(diag, sub, Eigen::ComputeEigenvectors);
  if (es.info() != Eigen::Success)
    throw std::runtime_error("gaussJacobi: eigen-decomposition failed");

  Rule r;
  r.x = es.eigenvalues();                                    // ascending
  r.w = (es.eigenvectors().row(0).transpose().array().square() * mu0(a, b)).matrix();
  return r;
}

Rule gaussLegendre(int n) { return gaussJacobi(n, 0.0, 0.0); }

const Rule& gaussLegendreCached(int n) {
  static std::map<int, Rule> cache;
  auto it = cache.find(n);
  if (it == cache.end()) it = cache.emplace(n, gaussLegendre(n)).first;
  return it->second;
}

}  // namespace dgw::jacobi
