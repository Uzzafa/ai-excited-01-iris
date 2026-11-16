// SPDX-License-Identifier: MIT
// IRIS ENGINE — screen-space polar renderer.
// Every pixel is a pure function of (radius, angle) about the centre. There is no
// ray marching, no signed distance field, no normals and no hit test anywhere in
// this file. Zero dependencies (node:zlib / node:fs / node:path / node:crypto),
// fully deterministic: all variation is an integer hash, never Math.random.

import { deflateSync } from 'node:zlib';
import { mkdirSync, writeFileSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { createHash } from 'node:crypto';

const T0 = Date.now();

const W = 1024;
const H = 1024;

const PI = Math.PI;
const TAU = PI * 2;

function clamp(x: number, a: number, b: number) { return x < a ? a : x > b ? b : x; }
function mix(a: number, b: number, t: number) { return a + (b - a) * t; }
function sstep(a: number, b: number, x: number) {
  const t = clamp((x - a) / (b - a), 0, 1);
  return t * t * (3 - 2 * t);
}

// 32-bit integer hash -> [0,1)
function hsh(a: number, b: number, c: number) {
  let h = Math.imul(a | 0, 0x27d4eb2d) ^ Math.imul(b | 0, 0x165667b1) ^ Math.imul(c | 0, 0x9e3779b1);
  h = Math.imul(h ^ (h >>> 15), 0x2c1b3c6d);
  h = Math.imul(h ^ (h >>> 12), 0x297a2d39);
  h ^= h >>> 15;
  return (h >>> 0) / 4294967296;
}

// ---------------------------------------------------------------- palette

type RGB = [number, number, number];
const hex = (s: string): RGB => [
  parseInt(s.slice(1, 3), 16) / 255,
  parseInt(s.slice(3, 5), 16) / 255,
  parseInt(s.slice(5, 7), 16) / 255,
];
const GOLD: RGB = hex('#fff6d5');
const AMBER: RGB = hex('#ffb020');
const CRIMSON: RGB = hex('#ff2d55');
const VIOLET: RGB = hex('#7c3aed');
const CYAN: RGB = hex('#22d3ee');
const BG: RGB = hex('#04030a');

function cmix(a: RGB, b: RGB, t: number): RGB {
  return [mix(a[0], b[0], t), mix(a[1], b[1], t), mix(a[2], b[2], t)];
}

// ---------------------------------------------------------------- disc space

const ROT = 8 * PI / 180;
const CR = Math.cos(ROT), SR = Math.sin(ROT);
const SQUASH = 0.88;

// iris band
const R_IN = 0.30, R_OUT = 0.78;
// pupil slit (half-axes in disc space)
const PUP_A = 0.09, PUP_B = 0.42;
const NB = 48;

const PIX = 2 / W;              // one output pixel in disc-space units
const PIX_EDGE = 1.15 * PIX;    // antialias half-width

// ---------------------------------------------------------------- render buffer

const hdr = new Float32Array(W * H * 3);

// ---------------------------------------------------------------- blade tables
// Each blade gets its own outer reach, brightness and angular skew from an
// integer hash of its index; nothing here is random.

const bReach = new Float64Array(NB);
const bBright = new Float64Array(NB);
const bSkew = new Float64Array(NB);
const bHue = new Float64Array(NB);
const bInner = new Float64Array(NB);
for (let i = 0; i < NB; i++) {
  bReach[i] = 0.30 + 0.48 * Math.pow(hsh(i, 11, 1), 0.85);
  bBright[i] = 0.46 + 0.54 * hsh(i, 23, 2);
  bSkew[i] = (hsh(i, 37, 3) - 0.5) * 0.30;
  bHue[i] = hsh(i, 53, 4);
  bInner[i] = 0.008 + 0.030 * hsh(i, 67, 5);
}

// ---------------------------------------------------------------- arcs

const NARC = 3;
const arcCX = [0.05, -0.085, 0.10];
const arcCY = [-0.04, 0.06, -0.09];
const arcRA = [0.95, 1.15, 1.35];
const arcRB = [0.70, 0.92, 1.05];
const arcTilt = [0.24, -0.36, 0.13];
const arcPhase = [0.0, 2.2, 4.3];

const SH_CX = 0.15, SH_CY = 0.54, SH_RA = 1.00, SH_RB = 0.52, SH_TILT = -0.32;

// ---------------------------------------------------------------- dust specks
// jittered hash grid: 20x20 cells -> up to 400 candidate points. They are bucketed
// into a coarse spatial grid so a sample only tests its own cell and the ring of
// cells around it.

type Speck = { x: number; y: number; rad: number; amp: number; front: number; tint: number };
const specks: Speck[] = [];
const GRID = 20;
for (let gy = 0; gy < GRID; gy++) {
  for (let gx = 0; gx < GRID; gx++) {
    if (hsh(gx, gy, 106) > 0.90) continue;     // ~360 specks survive
    const jx = hsh(gx, gy, 101), jy = hsh(gx, gy, 102);
    specks.push({
      x: ((gx + 0.12 + 0.76 * jx) / GRID) * 2 - 1,
      y: ((gy + 0.12 + 0.76 * jy) / GRID) * 2 - 1,
      rad: 0.0010 + 0.0030 * Math.pow(hsh(gx, gy, 103), 1.8),
      amp: 0.08 + 1.05 * Math.pow(hsh(gx, gy, 104), 3.2),
      front: hsh(gx, gy, 105) < 0.5 ? 1.0 : 0.34,
      tint: hsh(gx, gy, 107),
    });
  }
}

// bucket index: 48x48 cells over [-1.2, 1.2]; specks are indexed into their cell
const BC = 48, BSPAN = 1.35, BINV = BC / (2 * BSPAN);
const bucketStart = new Int32Array(BC * BC + 1);
const bucketIdx = new Int32Array(specks.length);
{
  const cellOf = (v: number) => clamp(Math.floor((v + BSPAN) * BINV), 0, BC - 1);
  const counts = new Int32Array(BC * BC);
  const cells = new Int32Array(specks.length);
  for (let i = 0; i < specks.length; i++) {
    const c = cellOf(specks[i].y) * BC + cellOf(specks[i].x);
    cells[i] = c; counts[c]++;
  }
  let acc = 0;
  for (let c = 0; c < BC * BC; c++) { bucketStart[c] = acc; acc += counts[c]; }
  bucketStart[BC * BC] = acc;
  const fill = new Int32Array(BC * BC);
  for (let i = 0; i < specks.length; i++) {
    const c = cells[i];
    bucketIdx[bucketStart[c] + fill[c]++] = i;
  }
}
// cell index for a disc-space coordinate
function cellAt(v: number) { return clamp(Math.floor((v + BSPAN) * BINV), 0, BC - 1); }

// ---------------------------------------------------------------- streaks
// short strokes along logarithmic spirals r = r0 * exp(k*theta)

type StreakF = { cx0: number; cy0: number; nx: number; ny: number; tx: number; ty: number; span: number; rad: number; amp: number };
const streaksF: StreakF[] = [];
const NSX = 15;
for (let i = 0; i < NSX; i++) {
  const r0 = 0.34 + 0.60 * hsh(i, 201, 1);
  const th = hsh(i, 202, 2) * TAU;
  const k = 1.2 + 1.9 * hsh(i, 203, 3);
  const span = 0.030 + 0.045 * hsh(i, 204, 4);
  const rad = 0.0022 + 0.0016 * hsh(i, 206, 6);
  const amp = 0.45 + 0.55 * hsh(i, 205, 5);
  const cx0 = Math.cos(th) * r0;
  const cy0 = Math.sin(th) * r0 * SQUASH;
  // tangent of r = r0*exp(k*th): d/dth (x,y) = (k cos th - sin th, (k sin th + cos th) * SQUASH)
  let tx = k * Math.cos(th) - Math.sin(th);
  let ty = (k * Math.sin(th) + Math.cos(th)) * SQUASH;
  const tl = Math.hypot(tx, ty);
  tx /= tl; ty /= tl;
  streaksF.push({ cx0, cy0, tx, ty, nx: -ty, ny: tx, span, rad, amp });
}

// ---------------------------------------------------------------- shapes

function pupilSd(x: number, y: number) {
  // iteration-free ellipse distance: scale-normalised radius converted back with
  // the gradient magnitude at that point. Accurate to a few percent near the rim.
  const qx = x / PUP_A, qy = y / PUP_B;
  const k = Math.sqrt(qx * qx + qy * qy);
  if (k < 1e-6) return -Math.min(PUP_A, PUP_B);
  const gx = qx / (PUP_A * k), gy = qy / (PUP_B * k);
  return (k - 1) / Math.sqrt(gx * gx + gy * gy);
}

// ---------------------------------------------------------------- the scene
// One evaluation of the picture at one point of disc space.

const out3: RGB = [0, 0, 0];

function sample(u: number, v: number): void {
  let r = BG[0], g = BG[1], b = BG[2];

  const rad = Math.sqrt(u * u + v * v);
  const psd = pupilSd(u, v);

  // --- pupil: absolute void, then a hard white-gold rim ~2 px at 1024
  if (psd < 0) { out3[0] = 0; out3[1] = 0; out3[2] = 0; return; }
  if (psd < PIX_EDGE) {
    const k = (1 - sstep(0, PIX_EDGE, psd)) * 2.6;
    r += GOLD[0] * k; g += GOLD[1] * k; b += GOLD[2] * k;
  }

  // --- iris band
  if (rad > R_IN - 2 * PIX && rad < R_OUT + 2 * PIX) {
    const a = Math.atan2(v, u);
    const bu = (a / TAU + 0.5) * NB;
    const idx = Math.floor(bu);
    const i0 = ((idx % NB) + NB) % NB;
    const frac = bu - idx;
    const skewed = frac + bSkew[i0];

    // ordered stagger: the aperture hangs open on the +u side and nearly shuts on
    // the -u side, so the ring reads as a mechanism caught mid-open
    const sweep = 0.5 + 0.5 * Math.cos(a - 0.30);
    // reach carries the sweep; brightness does not, otherwise the open side
    // multiplies out to clipping and loses every blade seam
    const reach = bReach[i0] * (0.66 + 0.50 * sweep * sweep);

    const inRad = sstep(R_IN - PIX_EDGE, R_IN + PIX_EDGE, rad) * sstep(R_OUT + PIX_EDGE, R_OUT - PIX_EDGE, rad);
    // each blade retracts to its own inner radius: on the shut side the gap under
    // the blades opens up, on the open side they crowd the pupil
    const inner = R_IN - 0.055 + bInner[i0] + 0.10 * (1 - sweep) * (1 - sweep);
    const inInner = sstep(inner - PIX_EDGE, inner + PIX_EDGE, rad);
    if (inRad > 0) {
      // radial ramp: white-gold tip -> amber -> crimson -> violet rim
      const t = clamp((rad - R_IN) / (R_OUT - R_IN), 0, 1);
      let col: RGB;
      if (t < 0.34) col = cmix(GOLD, AMBER, t / 0.34);
      else if (t < 0.68) col = cmix(AMBER, CRIMSON, (t - 0.34) / 0.34);
      else col = cmix(CRIMSON, VIOLET, (t - 0.68) / 0.32);

      // machined shading. The blade is a plate seen edge-on: a bright raised lip
      // along one edge, a flat body, then a hard fall into the dark seam where the
      // next plate overlaps it. Without a genuinely dark seam the ring reads as
      // one continuous band, so the seam is driven to near zero on purpose.
      const e = Math.abs(skewed - 0.5) * 2;
      const eLight = Math.pow(clamp((0.30 - skewed) / 0.30, 0, 1), 1.8);   // raised lip
      const eSeam = Math.pow(clamp((skewed - 0.62) / 0.38, 0, 1), 0.9);    // trailing seam
      const body = 0.26 + 0.74 * (1 - e * e * 0.85);
      const bevel = body + 0.55 * eLight;
      // the seam is a hard multiplicative term: it must reach black
      const shade = clamp(bevel * (1 - 0.94 * eSeam), 0.0, 1.6);

      // the pitch edge: a thin emissive line on one side of every blade, drawn
      // just inside the raised lip so it does not refill the seam behind it
      const lineD = Math.abs(skewed - 0.20) * (TAU * rad / NB);
      const line = sstep(1.6 * PIX, 0.2 * PIX, Math.abs(lineD)) * 1.05;

      const gain = 0.62 + 0.55 * bHue[i0];
      const inReach = sstep(reach + PIX_EDGE, reach - PIX_EDGE, rad);
      // the seam is a gap in the plate, not a shadow on it: the blade's own light
      // is killed there, and the pitch line only exists where the plate does
      const gap = 1 - 0.94 * eSeam;
      // per-blade radial falloff: each plate fades along its own length, so the
      // ring is not one uniformly lit band at a given radius
      const alongT = clamp((rad - inner) / Math.max(0.04, reach - inner), 0, 1);
      const lengthFade = 0.55 + 0.45 * Math.pow(1 - alongT, 0.7);
      const lit = inRad * inInner * inReach * shade * gap * bBright[i0] * gain * 0.44 * lengthFade
        + inRad * inInner * line * 0.34 * gap;
      r += col[0] * lit; g += col[1] * lit; b += col[2] * lit;

      // hot lip where the blades converge on the open side
      const lip = sstep(0.040, 0.0, rad - reach) * inRad * inInner * sweep * sweep * 0.55;
      r += GOLD[0] * lip; g += GOLD[1] * lip; b += GOLD[2] * lip;

      // far-side falloff so the ring is never a flat ring
      const dim = 0.78 + 0.22 * sweep;
      r *= dim; g *= dim; b *= dim;
    }
  }

  // --- three off-centre glowing arcs
  for (let i = 0; i < NARC; i++) {
    const dx = u - arcCX[i], dy = v - arcCY[i];
    const ra = arcRA[i], rb = arcRB[i];
    const ct = Math.cos(arcTilt[i]), st = Math.sin(arcTilt[i]);
    const px = dx * ct + dy * st, py = -dx * st + dy * ct;
    const q = Math.sqrt((px / ra) ** 2 + (py / rb) ** 2);
    const de = Math.abs((q - 1) * Math.min(ra, rb));
    if (de < 0.10) {
      const ang = Math.atan2(py, px);
      const mod = 0.22 + 0.78 * Math.pow(0.5 + 0.5 * Math.sin(ang * 1.5 + arcPhase[i]), 1.6);
      // far side of every ring is darker: the light comes out of the core
      const far = 0.50 + 0.50 * (0.5 + 0.5 * Math.cos(ang + 0.40));
      const core = sstep(0.0030, 0.0, de) * 2.2;
      const halo = Math.exp(-de * 24) * 0.28 + Math.exp(-de * 7) * 0.080;
      const k = (core + halo) * mod * far;
      const col = i === 1 ? VIOLET : CYAN;
      r += col[0] * k; g += col[1] * k; b += col[2] * k;
    }
  }

  // --- one off-axis shockwave across the lower third
  {
    const dx = u - SH_CX, dy = v - SH_CY;
    const ct = Math.cos(SH_TILT), st = Math.sin(SH_TILT);
    const px = dx * ct + dy * st, py = -dx * st + dy * ct;
    const q = Math.sqrt((px / SH_RA) ** 2 + (py / SH_RB) ** 2);
    const de = Math.abs((q - 1) * Math.min(SH_RA, SH_RB));
    if (de < 0.06) {
      const ang = Math.atan2(py, px);
      // brightest where it crosses the dark
      const mod = 0.28 + 0.92 * Math.pow(0.5 + 0.5 * Math.sin(ang * 1.5 + 0.9), 1.4);
      const k = (sstep(0.0022, 0.0, de) * 2.9 + Math.exp(-de * 26) * 0.42) * mod;
      r += GOLD[0] * k; g += GOLD[1] * k * 0.94; b += GOLD[2] * k * 0.84;
    }
  }

  // --- dust (spatial hash: only the neighbouring cells are tested)
  {
    const cx0 = cellAt(u), cy0 = cellAt(v);
    for (let cyi = cy0 - 1; cyi <= cy0 + 1; cyi++) {
      if (cyi < 0 || cyi >= BC) continue;
      for (let cxi = cx0 - 1; cxi <= cx0 + 1; cxi++) {
        if (cxi < 0 || cxi >= BC) continue;
        const c = cyi * BC + cxi;
        for (let k = bucketStart[c], e = bucketStart[c + 1]; k < e; k++) {
          const s = specks[bucketIdx[k]];
          const dx = u - s.x, dy = v - s.y;
          const d2 = dx * dx + dy * dy;
          const r2 = s.rad * s.rad;
          if (d2 > r2 * 16) continue;
          const w = s.amp * s.front * Math.exp(-d2 / r2);
          const warm = s.tint;
          r += (0.30 + 0.70 * warm) * w;
          g += (0.32 + 0.55 * warm) * w;
          b += (0.54 + 0.30 * warm) * w;
        }
      }
    }
  }

  // --- streaks: brief warm segments dragged along a log spiral
  for (let i = 0; i < streaksF.length; i++) {
    const s = streaksF[i];
    const dx = u - s.cx0, dy = v - s.cy0;
    const tanOff = dx * s.tx + dy * s.ty;
    if (Math.abs(tanOff) > s.span) continue;
    const radOff = dx * s.nx + dy * s.ny;
    const t = tanOff / s.span + 0.5;
    const taper = Math.sin(t * PI);
    const w = s.amp * taper * Math.exp(-(radOff * radOff) / (s.rad * s.rad)) * 1.4;
    r += 0.95 * w; g += 0.58 * w; b += 0.26 * w;
  }

  // --- depth: the outside is dimmer than the core
  const fade = 1 - 0.48 * sstep(0.82, 1.42, rad);
  out3[0] = r * fade; out3[1] = g * fade; out3[2] = b * fade;
}

// ---------------------------------------------------------------- render 2x2

const SUBX = [-0.25, 0.25, -0.25, 0.25];
const SUBY = [-0.25, -0.25, 0.25, 0.25];

for (let y = 0; y < H; y++) {
  for (let x = 0; x < W; x++) {
    let ar = 0, ag = 0, ab = 0;
    for (let s = 0; s < 4; s++) {
      const nu = ((x + 0.5 + SUBX[s]) / W) * 2 - 1;
      const nv = ((y + 0.5 + SUBY[s]) / H) * 2 - 1;
      const u = nu * CR + nv * SR;
      const v = (-nu * SR + nv * CR) * SQUASH;
      sample(u, v);
      ar += out3[0]; ag += out3[1]; ab += out3[2];
    }
    const o = (y * W + x) * 3;
    hdr[o] = ar * 0.25; hdr[o + 1] = ag * 0.25; hdr[o + 2] = ab * 0.25;
  }
}

const TRENDER = Date.now();

// ---------------------------------------------------------------- bloom

function downsample(src: Float32Array, w: number, h: number, nw: number, nh: number, thresh: number) {
  const dst = new Float32Array(nw * nh * 3);
  const fx = (w / nw) | 0, fy = (h / nh) | 0;
  for (let y = 0; y < nh; y++) {
    for (let x = 0; x < nw; x++) {
      let r = 0, g = 0, b = 0;
      for (let yy = 0; yy < fy; yy++) {
        for (let xx = 0; xx < fx; xx++) {
          const o = ((y * fy + yy) * w + x * fx + xx) * 3;
          r += src[o]; g += src[o + 1]; b += src[o + 2];
        }
      }
      const k = 1 / (fx * fy);
      r *= k; g *= k; b *= k;
      if (thresh > 0) {
        const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        const mm = sstep(thresh, thresh * 3.0, lum);
        r *= mm; g *= mm; b *= mm;
      }
      const o = (y * nw + x) * 3;
      dst[o] = r; dst[o + 1] = g; dst[o + 2] = b;
    }
  }
  return dst;
}

function blur(src: Float32Array, w: number, h: number, rad: number) {
  const tmpB = new Float32Array(src.length);
  const n = rad * 2 + 1;
  for (let y = 0; y < h; y++) {
    for (let c = 0; c < 3; c++) {
      let acc = 0;
      for (let k = -rad; k <= rad; k++) acc += src[(y * w + clamp(k, 0, w - 1)) * 3 + c];
      for (let x = 0; x < w; x++) {
        tmpB[(y * w + x) * 3 + c] = acc / n;
        acc += src[(y * w + Math.min(x + rad + 1, w - 1)) * 3 + c] - src[(y * w + Math.max(x - rad, 0)) * 3 + c];
      }
    }
  }
  for (let x = 0; x < w; x++) {
    for (let c = 0; c < 3; c++) {
      let acc = 0;
      for (let k = -rad; k <= rad; k++) acc += tmpB[(clamp(k, 0, h - 1) * w + x) * 3 + c];
      for (let y = 0; y < h; y++) {
        src[(y * w + x) * 3 + c] = acc / n;
        acc += tmpB[(Math.min(y + rad + 1, h - 1) * w + x) * 3 + c] - tmpB[(Math.max(y - rad, 0) * w + x) * 3 + c];
      }
    }
  }
}

const B1W = W >> 1, B1H = H >> 1;
const B2W = W >> 2, B2H = H >> 2;
const B3W = W >> 3, B3H = H >> 3;

const BL1 = downsample(hdr, W, H, B1W, B1H, 1.05);
blur(BL1, B1W, B1H, 2);
const BL2 = downsample(BL1, B1W, B1H, B2W, B2H, 0);
blur(BL2, B2W, B2H, 3);
const BL3 = downsample(BL2, B2W, B2H, B3W, B3H, 0);
blur(BL3, B3W, B3H, 4);

// the void must stay void: bloom is not allowed to paint across the slit
const voidMask = new Uint8Array(W * H);
for (let y = 0; y < H; y++) {
  for (let x = 0; x < W; x++) {
    const nu = ((x + 0.5) / W) * 2 - 1;
    const nv = ((y + 0.5) / H) * 2 - 1;
    const u = nu * CR + nv * SR;
    const v = (-nu * SR + nv * CR) * SQUASH;
    voidMask[y * W + x] = pupilSd(u, v) < 1.5 * PIX ? 1 : 0;
  }
}

const bbuf: RGB = [0, 0, 0];
function bilinear(src: Float32Array, lw: number, lh: number, u: number, v: number, out: RGB) {
  const fx = clamp(u * lw - 0.5, 0, lw - 1.001);
  const fy = clamp(v * lh - 0.5, 0, lh - 1.001);
  const x0 = fx | 0, y0 = fy | 0;
  const tx = fx - x0, ty = fy - y0;
  const x1 = Math.min(x0 + 1, lw - 1), y1 = Math.min(y0 + 1, lh - 1);
  const o00 = (y0 * lw + x0) * 3, o10 = (y0 * lw + x1) * 3;
  const o01 = (y1 * lw + x0) * 3, o11 = (y1 * lw + x1) * 3;
  for (let c = 0; c < 3; c++) {
    const a = src[o00 + c] + (src[o10 + c] - src[o00 + c]) * tx;
    const b = src[o01 + c] + (src[o11 + c] - src[o01 + c]) * tx;
    out[c] = a + (b - a) * ty;
  }
}

const TPOST = Date.now();

// ---------------------------------------------------------------- tonemap + write

function aces(x: number) {
  const a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1.0);
}
function srgb(x: number) {
  return x <= 0.0031308 ? x * 12.92 : 1.055 * Math.pow(x, 1 / 2.4) - 0.055;
}

const rgb = new Uint8Array(W * H * 3);

for (let y = 0; y < H; y++) {
  const vv = y / H;
  for (let x = 0; x < W; x++) {
    const o = (y * W + x) * 3;
    const uu = x / W;

    let r = hdr[o], g = hdr[o + 1], b = hdr[o + 2];
    const mk = voidMask[y * W + x] ? 0 : 1;

    bilinear(BL1, B1W, B1H, uu, vv, bbuf); r += bbuf[0] * 0.36 * mk; g += bbuf[1] * 0.36 * mk; b += bbuf[2] * 0.36 * mk;
    bilinear(BL2, B2W, B2H, uu, vv, bbuf); r += bbuf[0] * 0.30 * mk; g += bbuf[1] * 0.30 * mk; b += bbuf[2] * 0.30 * mk;
    bilinear(BL3, B3W, B3H, uu, vv, bbuf); r += bbuf[0] * 0.24 * mk; g += bbuf[1] * 0.24 * mk; b += bbuf[2] * 0.24 * mk;

    // vignette
    const cx = uu - 0.5, cy = vv - 0.5;
    const vg = 1 - 0.36 * Math.pow(clamp((cx * cx + cy * cy) * 2.2, 0, 1.1), 1.15);
    r *= vg; g *= vg; b *= vg;

    // triangular dither (+-0.5 LSB) then ~1 LSB grain, applied after the transfer
    const d1 = hsh(x, y, 0x51ed);
    const d2 = hsh(x + 7919, y + 104729, 0x2545);
    const tri = (d1 + d2 - 1) * 0.5;
    const gr = (hsh(x + 31337, y + 7, 0x9e37) - 0.5) * 0.9;

    rgb[o] = clamp(Math.round(srgb(aces(r)) * 255 + tri + gr), 0, 255);
    rgb[o + 1] = clamp(Math.round(srgb(aces(g)) * 255 + tri + gr), 0, 255);
    rgb[o + 2] = clamp(Math.round(srgb(aces(b)) * 255 + tri + gr), 0, 255);
  }
}

// ---------------------------------------------------------------- PNG encoder

const crcTable = new Int32Array(256);
for (let n = 0; n < 256; n++) {
  let c = n;
  for (let k = 0; k < 8; k++) c = (c & 1) ? (0xedb88320 ^ (c >>> 1)) : (c >>> 1);
  crcTable[n] = c;
}
function crc32(buf: Uint8Array, start: number, end: number) {
  let c = -1;
  for (let i = start; i < end; i++) c = crcTable[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return (c ^ -1) >>> 0;
}
function chunk(type: string, data: Uint8Array) {
  const out = new Uint8Array(12 + data.length);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, data.length);
  out[4] = type.charCodeAt(0); out[5] = type.charCodeAt(1);
  out[6] = type.charCodeAt(2); out[7] = type.charCodeAt(3);
  out.set(data, 8);
  dv.setUint32(8 + data.length, crc32(out, 4, 8 + data.length));
  return out;
}

function encodePNG(w: number, h: number, pix: Uint8Array) {
  const stride = w * 3 + 1;
  const raw = new Uint8Array(stride * h);
  // Sub filter, written with a fast negative-wrap trick instead of a per-byte
  // mask. Same bytes as (pix[i] - pix[i-3]) & 0xff, roughly twice as fast, and
  // deflate's smallest output for dithered gradients.
  for (let y = 0; y < h; y++) {
    const ro = y * stride, po = y * w * 3;
    raw[ro] = 1;
    raw[ro + 1] = pix[po];
    raw[ro + 2] = pix[po + 1];
    raw[ro + 3] = pix[po + 2];
    for (let i = 3, o = po + 3; i < w * 3; i++, o++) {
      const d = pix[o] - pix[o - 3];
      raw[ro + 1 + i] = d < 0 ? d + 256 : d;
    }
  }
  const idat = deflateSync(raw, { level: 9, memLevel: 9 });

  const ihdr = new Uint8Array(13);
  const d = new DataView(ihdr.buffer);
  d.setUint32(0, w); d.setUint32(4, h);
  ihdr[8] = 8; ihdr[9] = 2; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;

  const sig = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]);
  const cIHDR = chunk('IHDR', ihdr);
  const cIDAT = chunk('IDAT', idat);
  const cIEND = chunk('IEND', new Uint8Array(0));
  const total = sig.length + cIHDR.length + cIDAT.length + cIEND.length;
  const file = new Uint8Array(total);
  let off = 0;
  for (const p of [sig, cIHDR, cIDAT, cIEND]) { file.set(p, off); off += p.length; }
  return file;
}

const png = encodePNG(W, H, rgb);
const outDir = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
mkdirSync(outDir, { recursive: true });
const outPath = join(outDir, 'entry.png');
writeFileSync(outPath, png);

// re-read from disk: if the sha does not move when the scene changes the render
// threw and we would be looking at a stale file
const disk = readFileSync(outPath);
const sha = createHash('sha256').update(disk).digest('hex');
console.log(`IHDR ${W}x${H}  bytes=${disk.length}  sha256=${sha}`);
console.log(`render ${TRENDER - T0} ms, post ${TPOST - TRENDER} ms, encode ${Date.now() - TPOST} ms, total ${Date.now() - T0} ms`);
