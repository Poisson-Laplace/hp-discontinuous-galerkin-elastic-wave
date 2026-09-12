#!/usr/bin/env python3
"""Quick-look renderer for the legacy-ASCII VTK files written by dgwave.

Only numpy and matplotlib are needed -- no vtk, no meshio. ParaView is the
better tool once you are steering a real simulation; this is for the tight
edit-build-look loop.

    python3 tools/quicklook.py out_mesh_hp.vtk
    python3 tools/quicklook.py out_mesh_hp.vtk --field pulse --save pulse.png
    python3 tools/quicklook.py out_mesh_hp.vtk --list
"""
import argparse
import sys

import numpy as np


def read_vtk(path):
    """Parse the subset of legacy VTK that dgwave emits."""
    with open(path) as fh:
        tok = fh.read().split()

    i, pts, tris, point_data, cell_data = 0, None, None, {}, {}
    n_pts = n_cells = 0
    section = None

    while i < len(tok):
        w = tok[i]
        if w == "POINTS":
            n_pts = int(tok[i + 1])
            vals = np.array(tok[i + 3: i + 3 + 3 * n_pts], dtype=float)
            pts = vals.reshape(n_pts, 3)[:, :2]
            i += 3 + 3 * n_pts
        elif w == "CELLS":
            n_cells, total = int(tok[i + 1]), int(tok[i + 2])
            vals = np.array(tok[i + 3: i + 3 + total], dtype=int)
            tris = vals.reshape(n_cells, 4)[:, 1:]
            i += 3 + total
        elif w == "CELL_TYPES":
            i += 2 + int(tok[i + 1])
        elif w == "POINT_DATA":
            section, i = "point", i + 2
        elif w == "CELL_DATA":
            section, i = "cell", i + 2
        elif w == "SCALARS":
            name = tok[i + 1]
            n = n_pts if section == "point" else n_cells
            j = i + 4                      # skip type, ncomp, LOOKUP_TABLE
            while tok[j] != "default":
                j += 1
            j += 1
            vals = np.array(tok[j: j + n], dtype=float)
            (point_data if section == "point" else cell_data)[name] = vals
            i = j + n
        else:
            i += 1

    if pts is None or tris is None:
        sys.exit(f"{path}: no POINTS/CELLS found")
    return pts, tris, point_data, cell_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vtk")
    ap.add_argument("--field", help="field to plot (default: first available)")
    ap.add_argument("--save", help="write to a PNG instead of showing a window")
    ap.add_argument("--list", action="store_true", help="list fields and exit")
    ap.add_argument("--cmap", default="RdBu_r")
    ap.add_argument("--edges", action="store_true",
                    help="overlay the sub-triangle wireframe")
    args = ap.parse_args()

    pts, tris, pd, cd = read_vtk(args.vtk)

    if args.list or (not pd and not cd):
        print(f"{len(pts)} points, {len(tris)} sub-triangles")
        print("  point fields:", ", ".join(pd) or "(none)")
        print("  cell  fields:", ", ".join(cd) or "(none)")
        return

    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    name = args.field or (next(iter(pd)) if pd else next(iter(cd)))
    T = mtri.Triangulation(pts[:, 0], pts[:, 1], tris)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    if name in pd:
        v = pd[name]
        lim = np.abs(v).max() or 1.0
        # gouraud interpolates within each sub-triangle but never across the
        # element boundary, so genuine DG jumps stay visible
        im = ax.tripcolor(T, v, shading="gouraud", cmap=args.cmap,
                          vmin=-lim, vmax=lim)
    elif name in cd:
        v = cd[name]
        im = ax.tripcolor(T, facecolors=v, cmap="viridis")
    else:
        sys.exit(f"no field named {name!r}; try --list")

    if args.edges:
        ax.triplot(T, color="k", lw=0.15, alpha=0.4)

    fig.colorbar(im, ax=ax, label=name)
    ax.set_aspect("equal")
    ax.set_title(f"{args.vtk} : {name}")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=130)
        print("wrote", args.save)
    else:
        plt.show()


if __name__ == "__main__":
    main()
