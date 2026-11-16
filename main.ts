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

// ---------------------------------------------------------------- render buffer

const hdr = new Float32Array(W * H * 3);

function addc(o: number, c: RGB, k: number) {
  hdr[o] += BG[0] + c[0] * k;
  hdr[o + 1] += BG[1] + c[1] * k;
  hdr[o + 2] += BG[2] + c[2] * k;
}

// ---------------------------------------------------------------- STAGE 1: flat disc

function discColor(r: number): RGB {
  const t = clamp((r - 0.30) / 0.48, 0, 1);
  return cmix(GOLD, CRIMSON, t);
}

for (let y = 0; y < H; y++) {
  const py = y + 0.5;
  for (let x = 0; x < W; x++) {
    const px = x + 0.5;
    const nu = (px / W) * 2 - 1;
    const nv = (py / H) * 2 - 1;
    const u = nu * CR + nv * SR;
    const v = (-nu * SR + nv * CR) * SQUASH;
    const r = Math.sqrt(u * u + v * v);
    const o = (y * W + x) * 3;
    hdr[o] = BG[0]; hdr[o + 1] = BG[1]; hdr[o + 2] = BG[2];
    if (r < 0.78) {
      const c = discColor(r);
      const shade = 0.62 * (1 - 0.42 * sstep(0.55, 0.78, r));
      addc(o, c, shade);
    }
  }
}

const TRENDER = Date.now();

// ---------------------------------------------------------------- tonemap

function aces(x: number) {
  const a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1.0);
}
function srgb(x: number) {
  return x <= 0.0031308 ? x * 12.92 : 1.055 * Math.pow(x, 1 / 2.4) - 0.055;
}

const rgb = new Uint8Array(W * H * 3);

for (let y = 0; y < H; y++) {
  for (let x = 0; x < W; x++) {
    const o = (y * W + x) * 3;
    const vg = 1 - 0.34 * Math.pow(clamp(((x / W - 0.5) ** 2 + (y / H - 0.5) ** 2) * 2.2, 0, 1.1), 1.15);
    for (let c = 0; c < 3; c++) {
      const lin = srgb(aces(hdr[o + c] * vg));
      const d1 = hsh(x, y, 0x51ed);
      const d2 = hsh(x + 7919, y + 104729, 0x2545);
      const tri = (d1 + d2 - 1) * 0.5;
      const gr = (hsh(x + 31337, y + 7, 0x9e37) - 0.5) * 0.9;
      rgb[o + c] = clamp(Math.round(lin * 255 + tri + gr), 0, 255);
    }
  }
}

const TPOST = Date.now();

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
  for (let y = 0; y < h; y++) {
    const ro = y * stride, po = y * w * 3;
    raw[ro] = 1; // Sub filter keeps the dithered gradients compressible
    for (let i = 0; i < w * 3; i++) {
      raw[ro + 1 + i] = (pix[po + i] - (i >= 3 ? pix[po + i - 3] : 0)) & 0xff;
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
