#pragma once
#include <array>
#include <vector>

#include "dgwave/RefTriangle.hpp"
#include "dgwave/Types.hpp"

namespace dgw {

/// Where a face sits in the domain. The physics layer decides what each tag
/// means (free surface, absorbing, ...); the mesh only records geometry.
enum BoundaryTag : int {
  kInterior = 0,
  kBottom   = 1,
  kRight    = 2,
  kTop      = 3,
  kLeft     = 4
};

/// Affine geometry of one triangle. Every quantity here is CONSTANT over the
/// element, which is what makes the modal mass matrix exactly J*I and lets the
/// metric terms come out of every quadrature loop.
struct ElementGeom {
  Real J = 0.0;  ///< Jacobian of the reference->physical map. Area = 2*J.

  // Inverse metric terms: dr/dx, dr/dy, ds/dx, ds/dy.
  Real drdx = 0.0, drdy = 0.0, dsdx = 0.0, dsdy = 0.0;

  std::array<Real, 3> faceLength{};        ///< physical edge lengths
  std::array<Vec2, 3> faceNormal{};        ///< physical outward unit normals
  std::array<Real, 3> fscale{};            ///< faceLength / (2*J), the surface
                                           ///< term's geometric factor

  Real inradius = 0.0;  ///< area / semiperimeter; drives the CFL estimate
};

/// A face shared by at most two elements.
///
/// elem[1] < 0 marks a boundary face. The two sides traverse the face in
/// opposite directions, so quadrature point q on side 0 coincides with
/// FaceGeom::mirrorIndex(nfq, q) on side 1 — an exact index flip, never an
/// interpolation.
struct Face {
  std::array<int, 2> elem{-1, -1};
  std::array<int, 2> face{-1, -1};  ///< local face id (0..2) on each side
  int tag = kInterior;

  bool isBoundary() const { return elem[1] < 0; }
};

/// Unstructured triangular mesh with per-element polynomial degree.
class Mesh {
 public:
  /// Build from raw vertex and triangle arrays. Triangles whose vertices are
  /// ordered clockwise are flipped so that every Jacobian comes out positive.
  Mesh(std::vector<Vec2> vertices, std::vector<std::array<int, 3>> elements,
       int uniformDegree);

  int numVertices() const { return static_cast<int>(vertices_.size()); }
  int numElements() const { return static_cast<int>(elements_.size()); }
  int numFaces() const { return static_cast<int>(faces_.size()); }

  const std::vector<Vec2>& vertices() const { return vertices_; }
  const std::array<int, 3>& element(int e) const { return elements_[e]; }
  const ElementGeom& geom(int e) const { return geom_[e]; }

  const std::vector<Face>& faces() const { return faces_; }
  const Face& face(int f) const { return faces_[f]; }
  /// Global face index of local face `lf` of element `e`.
  int elemFace(int e, int lf) const { return elemFace_[e][lf]; }

  int degree(int e) const { return degree_[e]; }
  void setDegree(int e, int p);
  int maxDegree() const;

  /// Face quadrature size for a shared face, taken from the higher of the two
  /// adjacent degrees so that both traces are integrated exactly.
  int faceQuadSize(int f, int overIntegration = 1) const;

  /// Reference (r,s) -> physical (x,y) on element e.
  Vec2 toPhysical(int e, Real r, Real s) const;

  /// Total area, computed as the sum of 2*J.
  Real area() const;

  /// Structured triangulation of [x0,x1] x [y0,y1] with nx*ny cells, each cell
  /// split into two triangles with the diagonal alternating between cells to
  /// avoid a directional bias in the mesh.
  static Mesh rectangle(Real x0, Real x1, Real y0, Real y1, int nx, int ny,
                        int uniformDegree);

 private:
  void buildGeometry();
  void buildConnectivity();

  std::vector<Vec2> vertices_;
  std::vector<std::array<int, 3>> elements_;
  std::vector<int> degree_;
  std::vector<ElementGeom> geom_;
  std::vector<Face> faces_;
  std::vector<std::array<int, 3>> elemFace_;
};

}  // namespace dgw
