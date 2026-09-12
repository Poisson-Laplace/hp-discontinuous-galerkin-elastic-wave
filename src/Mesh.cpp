#include "dgwave/Mesh.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <utility>

namespace dgw {

namespace {

// Affine map from the reference triangle to a physical one:
//
//   X(r,s) = -(r+s)/2 * X0 + (1+r)/2 * X1 + (1+s)/2 * X2
//
// so that (-1,-1) -> X0, (1,-1) -> X1, (-1,1) -> X2. The derivatives are
// constant:  dX/dr = (X1-X0)/2,  dX/ds = (X2-X0)/2.
//
// Local faces follow the vertex cycle: f0 = v0->v1, f1 = v1->v2, f2 = v2->v0,
// matching FaceGeom on the reference element.
constexpr int kFaceV0[3] = {0, 1, 2};
constexpr int kFaceV1[3] = {1, 2, 0};

}  // namespace

Mesh::Mesh(std::vector<Vec2> vertices, std::vector<std::array<int, 3>> elements,
           int uniformDegree)
    : vertices_(std::move(vertices)), elements_(std::move(elements)) {
  if (uniformDegree < 0 || uniformDegree > kMaxDegree)
    throw std::invalid_argument("Mesh: degree out of range");
  for (const auto& e : elements_)
    for (int v : e)
      if (v < 0 || v >= numVertices())
        throw std::out_of_range("Mesh: vertex index out of range");

  degree_.assign(elements_.size(), uniformDegree);
  buildGeometry();
  buildConnectivity();
}

void Mesh::buildGeometry() {
  geom_.assign(elements_.size(), ElementGeom{});

  for (std::size_t e = 0; e < elements_.size(); ++e) {
    auto& tri = elements_[e];
    const Vec2 X0 = vertices_[tri[0]];
    const Vec2 X1 = vertices_[tri[1]];
    const Vec2 X2 = vertices_[tri[2]];

    Real signedArea2 = (X1[0] - X0[0]) * (X2[1] - X0[1]) -
                       (X2[0] - X0[0]) * (X1[1] - X0[1]);
    if (std::abs(signedArea2) < 1e-30)
      throw std::runtime_error("Mesh: degenerate triangle (zero area)");
    if (signedArea2 < 0.0) {
      // Restore counter-clockwise ordering so that the outward normals below,
      // and the reference-element normals, agree.
      std::swap(tri[1], tri[2]);
      signedArea2 = -signedArea2;
    }

    const Vec2 A = vertices_[tri[0]];
    const Vec2 B = vertices_[tri[1]];
    const Vec2 C = vertices_[tri[2]];

    const Real dxdr = 0.5 * (B[0] - A[0]), dxds = 0.5 * (C[0] - A[0]);
    const Real dydr = 0.5 * (B[1] - A[1]), dyds = 0.5 * (C[1] - A[1]);

    ElementGeom g;
    g.J = dxdr * dyds - dxds * dydr;  // = area / 2
    g.drdx =  dyds / g.J;
    g.drdy = -dxds / g.J;
    g.dsdx = -dydr / g.J;
    g.dsdy =  dxdr / g.J;

    Real perimeter = 0.0;
    for (int f = 0; f < 3; ++f) {
      const Vec2 P = vertices_[tri[kFaceV0[f]]];
      const Vec2 Q = vertices_[tri[kFaceV1[f]]];
      const Real dx = Q[0] - P[0], dy = Q[1] - P[1];
      const Real L = std::hypot(dx, dy);
      g.faceLength[f] = L;
      // Outward normal for a counter-clockwise vertex cycle.
      g.faceNormal[f] = Vec2(dy / L, -dx / L);
      g.fscale[f] = L / (2.0 * g.J);
      perimeter += L;
    }
    g.inradius = (2.0 * g.J) / perimeter;  // area / semiperimeter

    geom_[e] = g;
  }
}

void Mesh::buildConnectivity() {
  faces_.clear();
  elemFace_.assign(elements_.size(), {-1, -1, -1});

  // Key a face by its sorted vertex pair; the second visit closes the pair.
  std::map<std::pair<int, int>, int> seen;

  for (std::size_t e = 0; e < elements_.size(); ++e) {
    for (int lf = 0; lf < 3; ++lf) {
      const int a = elements_[e][kFaceV0[lf]];
      const int b = elements_[e][kFaceV1[lf]];
      const auto key = std::minmax(a, b);

      auto it = seen.find(key);
      if (it == seen.end()) {
        Face fc;
        fc.elem[0] = static_cast<int>(e);
        fc.face[0] = lf;
        faces_.push_back(fc);
        const int gid = static_cast<int>(faces_.size()) - 1;
        seen.emplace(key, gid);
        elemFace_[e][lf] = gid;
      } else {
        const int gid = it->second;
        if (faces_[gid].elem[1] >= 0)
          throw std::runtime_error("Mesh: face shared by more than 2 elements");
        faces_[gid].elem[1] = static_cast<int>(e);
        faces_[gid].face[1] = lf;
        elemFace_[e][lf] = gid;
      }
    }
  }

  // Geometric boundary tags. Anything not on the bounding box keeps kInterior,
  // which the physics layer can still treat as a boundary via isBoundary().
  Real xmin = vertices_[0][0], xmax = xmin;
  Real ymin = vertices_[0][1], ymax = ymin;
  for (const auto& v : vertices_) {
    xmin = std::min(xmin, v[0]); xmax = std::max(xmax, v[0]);
    ymin = std::min(ymin, v[1]); ymax = std::max(ymax, v[1]);
  }
  const Real tol = 1e-9 * std::max(xmax - xmin, ymax - ymin);

  for (auto& fc : faces_) {
    if (!fc.isBoundary()) continue;
    const auto& tri = elements_[fc.elem[0]];
    const Vec2 P = vertices_[tri[kFaceV0[fc.face[0]]]];
    const Vec2 Q = vertices_[tri[kFaceV1[fc.face[0]]]];
    if (std::abs(P[1] - ymin) < tol && std::abs(Q[1] - ymin) < tol)
      fc.tag = kBottom;
    else if (std::abs(P[1] - ymax) < tol && std::abs(Q[1] - ymax) < tol)
      fc.tag = kTop;
    else if (std::abs(P[0] - xmin) < tol && std::abs(Q[0] - xmin) < tol)
      fc.tag = kLeft;
    else if (std::abs(P[0] - xmax) < tol && std::abs(Q[0] - xmax) < tol)
      fc.tag = kRight;
  }
}

void Mesh::setDegree(int e, int p) {
  if (p < 0 || p > kMaxDegree) throw std::invalid_argument("setDegree: bad p");
  degree_.at(static_cast<std::size_t>(e)) = p;
}

int Mesh::maxDegree() const {
  return *std::max_element(degree_.begin(), degree_.end());
}

int Mesh::faceQuadSize(int f, int overIntegration) const {
  const Face& fc = faces_.at(static_cast<std::size_t>(f));
  int p = degree_[fc.elem[0]];
  if (!fc.isBoundary()) p = std::max(p, degree_[fc.elem[1]]);
  return p + 1 + overIntegration;
}

Vec2 Mesh::toPhysical(int e, Real r, Real s) const {
  const auto& tri = elements_[e];
  const Real l0 = -0.5 * (r + s), l1 = 0.5 * (1.0 + r), l2 = 0.5 * (1.0 + s);
  return l0 * vertices_[tri[0]] + l1 * vertices_[tri[1]] + l2 * vertices_[tri[2]];
}

Real Mesh::area() const {
  Real a = 0.0;
  for (const auto& g : geom_) a += 2.0 * g.J;
  return a;
}

Mesh Mesh::rectangle(Real x0, Real x1, Real y0, Real y1, int nx, int ny,
                     int uniformDegree) {
  if (nx < 1 || ny < 1) throw std::invalid_argument("rectangle: nx, ny >= 1");

  std::vector<Vec2> verts;
  verts.reserve(static_cast<std::size_t>((nx + 1) * (ny + 1)));
  for (int j = 0; j <= ny; ++j)
    for (int i = 0; i <= nx; ++i)
      verts.emplace_back(x0 + (x1 - x0) * i / nx, y0 + (y1 - y0) * j / ny);

  auto vid = [nx](int i, int j) { return j * (nx + 1) + i; };

  std::vector<std::array<int, 3>> tris;
  tris.reserve(static_cast<std::size_t>(2 * nx * ny));
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      const int v00 = vid(i, j),     v10 = vid(i + 1, j);
      const int v01 = vid(i, j + 1), v11 = vid(i + 1, j + 1);
      if ((i + j) % 2 == 0) {  // diagonal v00-v11
        tris.push_back({v00, v10, v11});
        tris.push_back({v00, v11, v01});
      } else {                 // diagonal v10-v01, alternated to kill the bias
        tris.push_back({v00, v10, v01});
        tris.push_back({v10, v11, v01});
      }
    }
  }
  return Mesh(std::move(verts), std::move(tris), uniformDegree);
}

}  // namespace dgw
