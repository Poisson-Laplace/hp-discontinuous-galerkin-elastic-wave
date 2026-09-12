#include "dgwave/Dubiner.hpp"

#include <cmath>
#include <map>
#include <stdexcept>

#include "dgwave/Jacobi.hpp"

namespace dgw::dubiner {

const std::vector<ModeIdx>& modes(int p) {
  if (p < 0 || p > kMaxDegree) throw std::invalid_argument("dubiner::modes: bad p");
  static std::map<int, std::vector<ModeIdx>> cache;
  auto it = cache.find(p);
  if (it != cache.end()) return it->second;

  std::vector<ModeIdx> m;
  m.reserve(static_cast<std::size_t>(numModes(p)));
  for (int d = 0; d <= p; ++d)          // graded by total degree -> hierarchical
    for (int i = 0; i <= d; ++i) m.push_back(ModeIdx{i, d - i});

  return cache.emplace(p, std::move(m)).first->second;
}

void rsToab(Real r, Real s, Real& a, Real& b) {
  const Real den = 1.0 - s;
  a = (std::abs(den) > 1e-12) ? (2.0 * (1.0 + r) / den - 1.0) : -1.0;
  b = s;
}

Real eval(Real r, Real s, int i, int j) {
  Real a, b;
  rsToab(r, s, a, b);
  const Real h1 = jacobi::P(a, 0.0, 0.0, i);
  const Real h2 = jacobi::P(b, 2.0 * i + 1.0, 0.0, j);
  return std::sqrt(2.0) * h1 * h2 * std::pow(1.0 - b, i);
}

void grad(Real r, Real s, int i, int j, Real& dpsi_dr, Real& dpsi_ds) {
  Real a, b;
  rsToab(r, s, a, b);

  const Real fa  = jacobi::P (a, 0.0, 0.0, i);
  const Real dfa = jacobi::dP(a, 0.0, 0.0, i);
  const Real gb  = jacobi::P (b, 2.0 * i + 1.0, 0.0, j);
  const Real dgb = jacobi::dP(b, 2.0 * i + 1.0, 0.0, j);

  const Real hb = 0.5 * (1.0 - b);

  // d/dr : only the a-coordinate depends on r, with da/dr = 2/(1-b).
  Real dr = dfa * gb;
  if (i > 0) dr *= std::pow(hb, i - 1);

  // d/ds : a depends on s as well, da/ds = (1+a)/(1-b).
  Real ds = dfa * (gb * 0.5 * (1.0 + a));
  if (i > 0) ds *= std::pow(hb, i - 1);

  Real tmp = dgb * std::pow(hb, i);
  if (i > 0) tmp -= 0.5 * i * gb * std::pow(hb, i - 1);
  ds += fa * tmp;

  // Fold in the sqrt(2) * 2^i normalisation of the collapsed factors.
  const Real scale = std::pow(2.0, i + 0.5);
  dpsi_dr = scale * dr;
  dpsi_ds = scale * ds;
}

}  // namespace dgw::dubiner
