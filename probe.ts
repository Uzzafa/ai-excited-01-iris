// SPDX-License-Identifier: MIT
// STEP 0 — prove the write path: 32x32 test card, sha256 read back from disk.

import { deflateSync } from 'node:zlib';
import { mkdirSync, writeFileSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { createHash } from 'node:crypto';

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
    raw[ro] = 1;
    for (let i = 0; i < w * 3; i++) raw[ro + 1 + i] = (pix[po + i] - (i >= 3 ? pix[po + i - 3] : 0)) & 0xff;
  }
  const idat = deflateSync(raw, { level: 9, memLevel: 9 });
  const ihdr = new Uint8Array(13);
  const d = new DataView(ihdr.buffer);
  d.setUint32(0, w); d.setUint32(4, h);
  ihdr[8] = 8; ihdr[9] = 2;
  const sig = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]);
  const parts = [sig, chunk('IHDR', ihdr), chunk('IDAT', idat), chunk('IEND', new Uint8Array(0))];
  const total = parts.reduce((s, p) => s + p.length, 0);
  const file = new Uint8Array(total);
  let off = 0;
  for (const p of parts) { file.set(p, off); off += p.length; }
  return file;
}

const outDir = process.env.MAKEFUN_OUTPUT_DIR ?? 'out';
mkdirSync(outDir, { recursive: true });
const path = join(outDir, 'probe32.png');

function card(colour: [number, number, number]) {
  const w = 32, h = 32;
  const px = new Uint8Array(w * h * 3);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const o = (y * w + x) * 3;
    const inner = x > 8 && x < 24 && y > 8 && y < 24;
    const v = inner ? colour : [4, 3, 10] as [number, number, number];
    px[o] = v[0]; px[o + 1] = v[1]; px[o + 2] = v[2];
  }
  return px;
}

function report(tag: string) {
  const disk = readFileSync(path);
  const sha = createHash('sha256').update(disk).digest('hex');
  console.log(`${tag}  bytes=${disk.length}  sha256=${sha}`);
  return sha;
}

writeFileSync(path, encodePNG(32, 32, card([255, 246, 213])));
const a = report('write#1 gold');
writeFileSync(path, encodePNG(32, 32, card([255, 45, 85])));
const b = report('write#2 crimson');
console.log(a !== b ? 'PASS: write path live, sha changed' : 'FAIL: stale file');
