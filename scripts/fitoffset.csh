#!/usr/bin/env python3
import sys, math, tempfile, subprocess
from pathlib import Path

if len(sys.argv) not in (4, 5):
    sys.exit("Usage: fitoffset.csh npar_rng npar_azi xcorr.dat [SNR]")

nr, na = int(sys.argv[1]), int(sys.argv[2])
src = Path(sys.argv[3])
snr = float(sys.argv[4]) if len(sys.argv) == 5 else 20.0

if nr not in (1,2,3) or na not in (1,2,3):
    sys.exit("ERROR: npar must be 1, 2, or 3")

rows = []
for line in src.read_text().splitlines():
    p = line.split()
    if len(p) >= 5:
        x, ro, y, ao, q = map(float, p[:5])
        if q > snr:
            rows.append((x, y, ro, ao))

if len(rows) < 8:
    sys.exit(f"FAILED - only {len(rows)} points above SNR {snr}")

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    rf, af = td/"r.xyz", td/"a.xyz"
    rf.write_text("".join(f"{x} {y} {ro}\n" for x,y,ro,ao in rows))
    af.write_text("".join(f"{x} {y} {ao}\n" for x,y,ro,ao in rows))

    def fit(path, n):
        out = subprocess.check_output(
            ["gmt","trend2d",str(path),"-Fp",f"-N{n}+r"],
            text=True
        ).split()
        c = [float(v) for v in out]
        return (c + [0.0,0.0,0.0])[:3]

    rc = fit(rf, nr)
    ac = fit(af, na)
    info = subprocess.check_output(
        ["gmt","info",str(rf),"-C"], text=True
    ).split()
    xmin,xmax,ymin,ymax = map(float, info[:4])

dx, dy = xmax-xmin, ymax-ymin
if dx == 0 or dy == 0:
    sys.exit("ERROR: zero coordinate span")

def convert(c):
    c0,c1,c2 = c
    intercept = c0 - c1*(xmax+xmin)/dx - c2*(ymax+ymin)/dy
    sx = c1*2.0/dx
    sy = c2*2.0/dy
    return intercept, sx, sy

r, sr, asr = convert(rc)
a, sa, asa = convert(ac)

if not all(math.isfinite(v) for v in (r,sr,asr,a,sa,asa)):
    sys.exit("ERROR: non-finite fitoffset result")

ri, ai = math.floor(r), math.floor(a)

print(f"rshift = {ri}")
print(f"sub_int_r = {r-ri:.9f}")
print(f"stretch_r = {sr:.12g}")
print(f"a_stretch_r = {asr:.12g}")
print(f"ashift = {ai}")
print(f"sub_int_a = {a-ai:.9f}")
print(f"stretch_a = {sa:.12g}")
print(f"a_stretch_a = {asa:.12g}")
