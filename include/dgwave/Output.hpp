#pragma once
#include <array>
#include <string>
#include <vector>

#include "dgwave/Mesh.hpp"
#include "dgwave/Types.hpp"

namespace dgw::io {

/// Per-element modal coefficients. Element e holds numModes(degree(e)) entries,
/// so a single ModalField carries a mixed-degree (hp) solution without any
/// padding.
using ModalField = std::vector<Vec>;

/// Uniform barycentric subdivision of the reference triangle into nsub^2
/// sub-triangles. This is how a degree-p modal field becomes something a
/// renderer can draw: sample the expansion on the lattice, then draw flat
/// sub-triangles. nsub ~ p+1 resolves the polynomial; below that you are
/// looking at a smoothed version of your own solution.
struct Lattice {
  Vec r, s;                                ///< lattice points on the ref triangle
  std::vector<std::array<int, 3>> tris;    ///< sub-triangle connectivity
  int numPoints() const { return static_cast<int>(r.size()); }
  int numTris() const { return static_cast<int>(tris.size()); }
};

/// Cached lattice for a given subdivision level (nsub >= 1).
const Lattice& lattice(int nsub);

/// L2-project an analytic function onto the mesh. With the orthonormal basis
/// the mass matrix is the identity, so this is one weighted transpose per
/// element and no solve.
template <typename F>
ModalField project(const Mesh& m, F&& f) {
  ModalField out(static_cast<std::size_t>(m.numElements()));
  for (int e = 0; e < m.numElements(); ++e) {
    const RefTriangle& ref = refTri(m.degree(e));
    Vec fq(ref.numVolQuad());
    for (int q = 0; q < ref.numVolQuad(); ++q) {
      const Vec2 X = m.toPhysical(e, ref.rq()[q], ref.sq()[q]);
      fq[q] = f(X[0], X[1]);
    }
    out[static_cast<std::size_t>(e)] =
        ref.V().transpose() * (ref.wq().asDiagonal() * fq);
  }
  return out;
}

/// L2 error of a modal field against an analytic function, computed on an
/// over-integrated rule so the error itself is not under-resolved.
template <typename F>
Real l2Error(const Mesh& m, const ModalField& u, F&& f) {
  Real acc = 0.0;
  for (int e = 0; e < m.numElements(); ++e) {
    const RefTriangle& ref = refTri(m.degree(e), 3);
    const Vec uq = ref.V() * u[static_cast<std::size_t>(e)];
    for (int q = 0; q < ref.numVolQuad(); ++q) {
      const Vec2 X = m.toPhysical(e, ref.rq()[q], ref.sq()[q]);
      const Real d = uq[q] - f(X[0], X[1]);
      acc += ref.wq()[q] * m.geom(e).J * d * d;
    }
  }
  return std::sqrt(acc);
}

/// Legacy-ASCII VTK unstructured grid writer.
///
/// Points are NOT merged across elements. The solution really is discontinuous,
/// so each element contributes its own copy of a shared vertex and the renderer
/// shows the true inter-element jump. A jump that fails to shrink under
/// refinement is a broken flux, not a rendering artefact.
class VtkWriter {
 public:
  VtkWriter(const Mesh& mesh, int nsub);

  /// Sample a modal field at every lattice point and attach it as point data.
  void addPointField(const std::string& name, const ModalField& u);

  /// Attach one value per element (degree, tag, velocity, ...) as cell data.
  void addCellField(const std::string& name, const std::vector<Real>& perElem);

  void write(const std::string& path) const;

 private:
  const Mesh& mesh_;
  int nsub_;
  std::vector<Real> px_, py_;
  std::vector<std::array<int, 3>> cells_;
  std::vector<int> cellElem_;  ///< owning element of each sub-triangle
  std::vector<std::pair<std::string, std::vector<Real>>> pointData_, cellData_;
};

}  // namespace dgw::io
