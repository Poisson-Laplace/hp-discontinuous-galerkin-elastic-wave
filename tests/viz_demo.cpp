// Writes VTK files for the pieces that exist so far, and prints the projection
// convergence table.
//
// There is no wavefield yet. What can be inspected is the geometry, the basis,
// and how well the discrete space represents a smooth function — which is the
// upper bound on anything the solver will ever achieve on that mesh.

#include <cmath>
#include <cstdio>
#include <vector>

#include "dgwave/Output.hpp"

using namespace dgw;

namespace {

/// A smooth pulse standing in for a wavefield snapshot, so the sampling path
/// gets exercised on something with real spatial structure.
Real pulse(Real x, Real y) {
  const Real x0 = 600.0, y0 = 500.0, sig = 130.0;
  const Real r = std::hypot(x - x0, y - y0);
  return std::exp(-(r * r) / (2 * sig * sig)) * std::cos(r / 55.0);
}

void writeMeshOverview() {
  Mesh m = Mesh::rectangle(0.0, 1500.0, 0.0, 900.0, 20, 12, 3);

  // A synthetic hp distribution: raise the degree near the pulse, drop it far
  // away. This is the shape a real adaptive criterion would produce.
  for (int e = 0; e < m.numElements(); ++e) {
    const Vec2 c = m.toPhysical(e, -1.0 / 3.0, -1.0 / 3.0);
    const Real d = std::hypot(c[0] - 600.0, c[1] - 500.0);
    m.setDegree(e, d < 250.0 ? 6 : (d < 500.0 ? 4 : 2));
  }

  std::vector<Real> deg(m.numElements()), inr(m.numElements());
  for (int e = 0; e < m.numElements(); ++e) {
    deg[e] = m.degree(e);
    inr[e] = m.geom(e).inradius;
  }

  io::VtkWriter w(m, 7);
  w.addCellField("degree", deg);
  w.addCellField("inradius", inr);
  w.addPointField("pulse", io::project(m, pulse));
  w.write("out_mesh_hp.vtk");
  std::printf("  out_mesh_hp.vtk        %d elements, mixed p (2/4/6)\n", m.numElements());
}

void writeBasisModes() {
  // One reference triangle, one mode at a time: the cheapest way to see that a
  // basis function is what you think it is.
  const std::vector<Vec2> v = {Vec2(-1, -1), Vec2(1, -1), Vec2(-1, 1)};
  Mesh single(v, {{{0, 1, 2}}}, 6);

  io::VtkWriter w(single, 40);
  const auto& md = dubiner::modes(6);
  for (int k : {0, 1, 2, 4, 7, 11, 17, 24}) {
    Vec c = Vec::Zero(static_cast<int>(md.size()));
    c[k] = 1.0;
    char name[64];
    std::snprintf(name, sizeof(name), "mode_%02d_i%d_j%d", k, md[k].i, md[k].j);
    w.addPointField(name, {c});
  }
  w.write("out_modes.vtk");
  std::printf("  out_modes.vtk          8 Dubiner modes on the reference triangle\n");
}

void writeUnderResolved() {
  // The same field on a mesh too coarse to represent it. The inter-element
  // jumps are the point: this is what an under-resolved DG solution looks like,
  // and it is exactly the signature to watch for in a real snapshot.
  Mesh m = Mesh::rectangle(0.0, 1500.0, 0.0, 900.0, 8, 5, 1);
  io::VtkWriter w(m, 4);
  w.addPointField("pulse", io::project(m, pulse));
  w.write("out_underresolved.vtk");
  std::printf("  out_underresolved.vtk  p=1 on a coarse mesh, jumps visible\n");
}

void convergenceTable() {
  std::puts("\nProjection error of a smooth pulse");
  std::puts("(the best any solver could do on this mesh)\n");

  std::puts("  p-refinement, fixed 10x6 mesh");
  std::puts("    p    L2 error     factor");
  Real prev = 0.0;
  for (int p = 1; p <= 8; ++p) {
    Mesh m = Mesh::rectangle(0.0, 1500.0, 0.0, 900.0, 10, 6, p);
    const Real err = io::l2Error(m, io::project(m, pulse), pulse);
    if (p == 1) std::printf("   %2d    %.3e        -\n", p, err);
    else        std::printf("   %2d    %.3e     %6.2fx\n", p, err, prev / err);
    prev = err;
  }

  std::puts("\n  h-refinement, fixed p=3");
  std::puts("    n      h       L2 error     rate");
  prev = 0.0;
  Real prevH = 0.0;
  for (int n : {4, 8, 16, 32}) {
    Mesh m = Mesh::rectangle(0.0, 1500.0, 0.0, 900.0, n, n, 3);
    const Real h = 1500.0 / n;
    const Real err = io::l2Error(m, io::project(m, pulse), pulse);
    if (prev == 0.0) std::printf("   %3d   %6.1f    %.3e       -\n", n, h, err);
    else std::printf("   %3d   %6.1f    %.3e    %5.2f\n", n, h, err,
                     std::log(prev / err) / std::log(prevH / h));
    prev = err;
    prevH = h;
  }
  std::puts("\n  p-refinement gains an order of magnitude per degree once the");
  std::puts("  pulse is resolved; h-refinement is stuck at the algebraic");
  std::puts("  rate p+1 = 4. That gap is the whole argument for high order.");
}

}  // namespace

int main() {
  std::puts("=====================================================================");
  std::puts(" dgwave - visual output");
  std::puts("=====================================================================\n");
  std::puts("Writing VTK files:");
  writeMeshOverview();
  writeBasisModes();
  writeUnderResolved();
  convergenceTable();
  std::puts("\nOpen the .vtk files in ParaView, or run:  python3 tools/quicklook.py out_mesh_hp.vtk");
  return 0;
}
