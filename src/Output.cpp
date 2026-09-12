#include "dgwave/Output.hpp"

#include <fstream>
#include <map>
#include <stdexcept>

#include "dgwave/Dubiner.hpp"

namespace dgw::io {

const Lattice& lattice(int nsub) {
  if (nsub < 1) throw std::invalid_argument("lattice: nsub >= 1");
  static std::map<int, Lattice> cache;
  auto it = cache.find(nsub);
  if (it != cache.end()) return it->second;

  Lattice L;
  const int n = nsub;
  const int np = (n + 1) * (n + 2) / 2;
  L.r.resize(np);
  L.s.resize(np);

  // Barycentric lattice: l1 = i/n toward v1, l2 = j/n toward v2.
  // X = l0*v0 + l1*v1 + l2*v2 with v0=(-1,-1), v1=(1,-1), v2=(-1,1).
  std::vector<int> rowOffset(n + 2, 0);
  int k = 0;
  for (int j = 0; j <= n; ++j) {
    rowOffset[j] = k;
    for (int i = 0; i + j <= n; ++i, ++k) {
      const Real l1 = static_cast<Real>(i) / n;
      const Real l2 = static_cast<Real>(j) / n;
      const Real l0 = 1.0 - l1 - l2;
      L.r[k] = -l0 + l1 - l2;
      L.s[k] = -l0 - l1 + l2;
    }
  }
  rowOffset[n + 1] = k;

  auto id = [&](int i, int j) { return rowOffset[j] + i; };
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i + j < n; ++i) {
      L.tris.push_back({id(i, j), id(i + 1, j), id(i, j + 1)});     // upward
      if (i + j < n - 1)
        L.tris.push_back({id(i + 1, j), id(i + 1, j + 1), id(i, j + 1)});
    }
  }
  return cache.emplace(nsub, std::move(L)).first->second;
}

VtkWriter::VtkWriter(const Mesh& mesh, int nsub) : mesh_(mesh), nsub_(nsub) {
  const Lattice& L = lattice(nsub_);
  const int npe = L.numPoints();

  px_.reserve(static_cast<std::size_t>(mesh_.numElements()) * npe);
  py_.reserve(static_cast<std::size_t>(mesh_.numElements()) * npe);

  for (int e = 0; e < mesh_.numElements(); ++e) {
    const int base = e * npe;
    for (int q = 0; q < npe; ++q) {
      const Vec2 X = mesh_.toPhysical(e, L.r[q], L.s[q]);
      px_.push_back(X[0]);
      py_.push_back(X[1]);
    }
    for (const auto& t : L.tris) {
      cells_.push_back({base + t[0], base + t[1], base + t[2]});
      cellElem_.push_back(e);
    }
  }
}

void VtkWriter::addPointField(const std::string& name, const ModalField& u) {
  if (static_cast<int>(u.size()) != mesh_.numElements())
    throw std::invalid_argument("addPointField: field size != element count");

  const Lattice& L = lattice(nsub_);
  const int npe = L.numPoints();
  std::vector<Real> vals(static_cast<std::size_t>(mesh_.numElements()) * npe);

  // Vandermonde from the modal basis to the lattice, cached per degree since a
  // uniform-p mesh would otherwise rebuild the same matrix thousands of times.
  std::map<int, Mat> vlat;
  for (int e = 0; e < mesh_.numElements(); ++e) {
    const int p = mesh_.degree(e);
    auto it = vlat.find(p);
    if (it == vlat.end()) {
      const auto& md = dubiner::modes(p);
      Mat V(npe, static_cast<int>(md.size()));
      for (int q = 0; q < npe; ++q)
        for (std::size_t c = 0; c < md.size(); ++c)
          V(q, static_cast<int>(c)) = dubiner::eval(L.r[q], L.s[q], md[c].i, md[c].j);
      it = vlat.emplace(p, std::move(V)).first;
    }
    const Vec sampled = it->second * u[static_cast<std::size_t>(e)];
    for (int q = 0; q < npe; ++q) vals[static_cast<std::size_t>(e) * npe + q] = sampled[q];
  }
  pointData_.emplace_back(name, std::move(vals));
}

void VtkWriter::addCellField(const std::string& name, const std::vector<Real>& perElem) {
  if (static_cast<int>(perElem.size()) != mesh_.numElements())
    throw std::invalid_argument("addCellField: field size != element count");
  std::vector<Real> vals(cells_.size());
  for (std::size_t c = 0; c < cells_.size(); ++c)
    vals[c] = perElem[static_cast<std::size_t>(cellElem_[c])];
  cellData_.emplace_back(name, std::move(vals));
}

void VtkWriter::write(const std::string& path) const {
  std::ofstream os(path);
  if (!os) throw std::runtime_error("VtkWriter: cannot open " + path);
  os.precision(9);

  os << "# vtk DataFile Version 3.0\ndgwave\nASCII\nDATASET UNSTRUCTURED_GRID\n";

  os << "POINTS " << px_.size() << " float\n";
  for (std::size_t i = 0; i < px_.size(); ++i)
    os << px_[i] << ' ' << py_[i] << " 0\n";

  os << "CELLS " << cells_.size() << ' ' << cells_.size() * 4 << '\n';
  for (const auto& c : cells_) os << "3 " << c[0] << ' ' << c[1] << ' ' << c[2] << '\n';

  os << "CELL_TYPES " << cells_.size() << '\n';
  for (std::size_t i = 0; i < cells_.size(); ++i) os << "5\n";  // VTK_TRIANGLE

  if (!pointData_.empty()) {
    os << "POINT_DATA " << px_.size() << '\n';
    for (const auto& [name, vals] : pointData_) {
      os << "SCALARS " << name << " float 1\nLOOKUP_TABLE default\n";
      for (Real v : vals) os << v << '\n';
    }
  }
  if (!cellData_.empty()) {
    os << "CELL_DATA " << cells_.size() << '\n';
    for (const auto& [name, vals] : cellData_) {
      os << "SCALARS " << name << " float 1\nLOOKUP_TABLE default\n";
      for (Real v : vals) os << v << '\n';
    }
  }
}

}  // namespace dgw::io
