#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
b57b_implied_outlier.py -- WHY one draw in the q=4 cell reports implied lambda-mismatch 5.6e-11.

b57's [T1c] printed, per cell, implied = |dlambda/dtheta|(theta*) * |theta* - atan(x_cubic)|.
Thirteen cells: medians 5.2e-17..1.2e-16, but the q=4 spread-5-75+ cell has a per-draw MAX of 5.6e-11.
Either (a) that draw's theta* is a DIFFERENT crossing than the cubic's leading positive root
    (multiple descents on one ray: the geometric grid's first bracket need not pair with rts[0]), or
(b) the finite-difference slope is wrong (h = 0.05*theta* brackets a second root), or
(c) the printed max is a different draw from the printed max of r_bisect.

This script re-runs ONLY that cell and, for the worst draw, prints the whole root list of Phi, the
whole descent set of the grid (how many disjoint intervals), the bracket index that the bisection used,
the slope and both implied variants (against EACH positive root).

DECISION RULE (registered before running):
  [W1] #(draws whose descent set has more than one contiguous block) -- if > 0, then (a) is live and
       the fix is to pair theta* with the ROOT IT BRACKETED, not with rts[0].
  [W2] for the worst draw, min over positive roots of |theta* - atan(r)| * slope: if that is <= 1e-14
       then the routes still agree and only my PAIRING was wrong (a bookkeeping defect of my own check,
       rule 9's 'check that prints zeros' family).
  [W3] if min over roots is still >> 1e-14, the cubic is genuinely not the boundary on that draw and
       lem:cubic needs a side condition -- report it, do not patch the claim quietly.
"""
import numpy as np

# Load b57's DEFINITIONS ONLY (its cell loop starts at the first banner print, which would re-run
# the whole 470-draw study). Cutting the source is verbatim reuse -- no hand-copied helper, which is
# the failure mode of retraction (p).
src = open("B/b57_descent_boundary_cubic.py", encoding="utf-8").read()
cut = src.index('print("=" * 130)')
m = {}
exec(compile(src[:cut], "b57-definitions", "exec"), m)

TH = m["TH"]
cs = [np.cos(np.deg2rad(x)) for x in (5, 25, 55, 80)]   # b57's q=4 cell, built inline there


def gen():
    L = m["lam_spd"](7)
    return L, m["blocks"](L, cs)


rows = []
for i in range(40):
    Lam, K = gen()
    s = m["setup"](Lam, K)
    rr = m["ray_L"](s)
    if rr is None:
        continue
    u = rr[0]
    d = m["ray_check"](s, u)
    A0, B2, v1, cf1 = s["A0"], s["B2"], s["v1"], s["cf1"]
    r = np.roots(d["coef"])
    pos = np.sort(r[np.abs(r.imag) < 1e-9].real)
    V = np.outer(m["SC"], v1) + np.outer(m["SN"], u)
    lam = m["crossings"](np.einsum("ij,jk,ik->i", V, A0, V), np.einsum("ij,jk,ik->i", V, B2, V))
    mask = lam < cf1
    blocks_n = int(np.sum(mask[1:] & ~mask[:-1])) + (1 if mask[0] else 0)
    dd = np.nan
    if len(pos) and np.isfinite(d["theta"]):
        dd = d["slop"] * min(abs(d["theta"] - np.arctan(rr2)) for rr2 in pos)
    rows.append(dict(i=i, theta=d["theta"], roots=pos, nblocks=blocks_n, npos=len(pos),
                     implied=d["implied"], mind=dd, slop=d["slop"],
                     xcubic=(pos[0] if len(pos) else np.nan),
                     xhi=d["xhi"], P=d["P"], D=d["D"], g=d["g"]))

worst = max(rows, key=lambda z: (z["implied"] if np.isfinite(z["implied"]) else -1))
print("=" * 120)
print("[W1] #(draws with more than one contiguous descent block on the ray) ="
      f" {sum(1 for z in rows if z['nblocks'] > 1)}  of {len(rows)}")
print("     #(draws with more than one positive root of Phi) ="
      f" {sum(1 for z in rows if z['npos'] > 1)}")
print("     #(draws with nblocks != npos) ="
      f" {sum(1 for z in rows if z['nblocks'] != z['npos'])}   <- pairing ambiguity")
print(f"[W2/W3] worst implied = {worst['implied']:.4e} at draw {worst['i']}")
for key in ("theta", "slop", "mind", "implied", "xhi", "P", "D", "g", "nblocks", "npos"):
    print(f"     {key:>8} = {worst[key]}")
print(f"     positive roots of Phi = {worst['roots']}")
print("     all implied values, sorted:")
vals = np.sort(np.array([z["implied"] for z in rows], dtype=float))
print("       " + np.array2string(vals, precision=2, max_line_width=118))
print("     min-over-roots implied values, sorted (the correctly paired statistic):")
vals2 = np.sort(np.array([z["mind"] for z in rows if np.isfinite(z["mind"])], dtype=float))
print("       " + np.array2string(vals2, precision=2, max_line_width=118))
print(f"     #(min-over-roots implied > 1e-14) = {int(np.sum(vals2 > 1e-14))}/{len(vals2)}")
