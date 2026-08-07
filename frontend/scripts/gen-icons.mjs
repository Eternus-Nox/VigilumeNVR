// Generates the "Vigilume Shield" app icon (a centered shield with an ascending
// signal of nested chevrons) as PNGs, with zero dependencies. Pure-math
// rasterizer (signed distance fields, 1px feathered AA) + minimal PNG encoder
// (RGBA8 *and* opaque RGB8, via node:zlib). Deterministic — no randomness.
//
// Outputs:
//   • Web PWA icons -> frontend/public/icons  (run as the `prebuild` npm script)
//   • iOS AppIcon   -> ios/Vigilume/Assets.xcassets/AppIcon.appiconset (opaque,
//     RGB, no alpha — the App Store rejects alpha on the 1024 master). Written
//     only when that directory exists, so the web build stays self-contained.
//
// Concept: a dark rounded field (radial #101724 -> #0b1017), a CENTERED shield —
// flat-topped with rounded shoulders, tapering to a rounded point — drawn as a
// sky-blue (#38bdf8) outline over a slightly-lifted dark interior (#0e1a2b).
// Inside, an ASCENDING SIGNAL: two nested upward chevrons (upper brighter
// #7dd3fc, lower #38bdf8 @0.75) with a bright catch-light dot (#bfe8fd) just
// above the top chevron. Reads as security + vigilant ascending light.
import { deflateSync } from 'node:zlib';
import { mkdirSync, writeFileSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const OUT_DIR = join(SCRIPT_DIR, '..', 'public', 'icons');
const IOS_DIR = join(
  SCRIPT_DIR, '..', '..',
  'ios', 'Vigilume', 'Assets.xcassets', 'AppIcon.appiconset',
);

// ---------- PNG encoding ----------

const CRC_TABLE = new Int32Array(256);
for (let n = 0; n < 256; n++) {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  CRC_TABLE[n] = c;
}

function crc32(buf) {
  let c = 0xffffffff;
  for (let i = 0; i < buf.length; i++) c = CRC_TABLE[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function chunk(type, data) {
  const out = Buffer.alloc(12 + data.length);
  out.writeUInt32BE(data.length, 0);
  out.write(type, 4, 'ascii');
  data.copy(out, 8);
  out.writeUInt32BE(crc32(out.subarray(4, 8 + data.length)), 8 + data.length);
  return out;
}

// Encode an RGBA canvas. When `opaque` is set, emit color type 2 (RGB, no alpha
// channel) so the file has provably no transparency — required for the iOS
// master. The canvas is composited over an opaque field, so channels are exact.
function encodePng(size, rgba, { opaque = false } = {}) {
  const channels = opaque ? 3 : 4;
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(size, 0);
  ihdr.writeUInt32BE(size, 4);
  ihdr[8] = 8; // bit depth
  ihdr[9] = opaque ? 2 : 6; // color type: 2 = RGB, 6 = RGBA
  const raw = Buffer.alloc(size * (1 + size * channels));
  for (let y = 0; y < size; y++) {
    const rowStart = y * (1 + size * channels);
    raw[rowStart] = 0; // filter: none
    for (let x = 0; x < size; x++) {
      const src = (y * size + x) * 4;
      const dst = rowStart + 1 + x * channels;
      raw[dst] = rgba[src];
      raw[dst + 1] = rgba[src + 1];
      raw[dst + 2] = rgba[src + 2];
      if (!opaque) raw[dst + 3] = rgba[src + 3];
    }
  }
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr),
    chunk('IDAT', deflateSync(raw, { level: 9 })),
    chunk('IEND', Buffer.alloc(0)),
  ]);
}

// ---------- rasterizer ----------

const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
// SDF value -> coverage with ~1px feather.
const aa = (sdf) => clamp01(0.5 - sdf);
const lerp = (a, b, t) => a + (b - a) * t;
const mix = (A, B, t) => [lerp(A[0], B[0], t), lerp(A[1], B[1], t), lerp(A[2], B[2], t)];

function sdRoundRect(x, y, cx, cy, hw, hh, r) {
  const qx = Math.abs(x - cx) - (hw - r);
  const qy = Math.abs(y - cy) - (hh - r);
  const ox = Math.max(qx, 0);
  const oy = Math.max(qy, 0);
  return Math.hypot(ox, oy) + Math.min(Math.max(qx, qy), 0) - r;
}

const sdCircle = (x, y, cx, cy, r) => Math.hypot(x - cx, y - cy) - r;

// Unsigned distance to a segment AB (a capsule spine). Subtract a half-width for
// a thick, round-capped stroke.
function sdSegment(px, py, ax, ay, bx, by) {
  const pax = px - ax, pay = py - ay;
  const bax = bx - ax, bay = by - ay;
  const denom = bax * bax + bay * bay;
  const h = denom > 0 ? clamp01((pax * bax + pay * bay) / denom) : 0;
  return Math.hypot(pax - bax * h, pay - bay * h);
}

// Signed distance to a closed polygon (IQ's method): magnitude is the distance
// to the nearest edge, sign is negative inside. Orientation-independent.
function sdPolygon(px, py, verts) {
  const n = verts.length;
  let d = Infinity;
  let s = 1;
  for (let i = 0, j = n - 1; i < n; j = i++) {
    const xi = verts[i][0], yi = verts[i][1];
    const xj = verts[j][0], yj = verts[j][1];
    const ex = xj - xi, ey = yj - yi;
    const wx = px - xi, wy = py - yi;
    const denom = ex * ex + ey * ey;
    const t = denom > 0 ? clamp01((wx * ex + wy * ey) / denom) : 0;
    const bx = wx - ex * t, by = wy - ey * t;
    d = Math.min(d, bx * bx + by * by);
    const c1 = py >= yi, c2 = py < yj, c3 = ex * wy > ey * wx;
    if ((c1 && c2 && c3) || (!c1 && !c2 && !c3)) s = -s;
  }
  return s * Math.sqrt(d);
}

class Canvas {
  constructor(size) {
    this.size = size;
    this.px = Buffer.alloc(size * size * 4); // transparent
  }
  // Paint a solid colour where coverage(x, y) returns alpha 0..1 (src-over).
  paint([r, g, b], coverage) {
    this.paintFn(() => [r, g, b], coverage);
  }
  // Paint with a position-dependent colour (gradients), src-over.
  paintFn(colorFn, coverage) {
    const { size, px } = this;
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const a = coverage(x + 0.5, y + 0.5);
        if (a <= 0) continue;
        const [r, g, b] = colorFn(x + 0.5, y + 0.5);
        const i = (y * size + x) * 4;
        const da = px[i + 3] / 255;
        const outA = a + da * (1 - a);
        if (outA <= 0) continue;
        px[i] = Math.round((r * a + px[i] * da * (1 - a)) / outA);
        px[i + 1] = Math.round((g * a + px[i + 1] * da * (1 - a)) / outA);
        px[i + 2] = Math.round((b * a + px[i + 2] * da * (1 - a)) / outA);
        px[i + 3] = Math.round(outA * 255);
      }
    }
  }
}

// ---------- palette ----------

const FIELD_CTR = [16, 23, 36];    // #101724 — radial centre (slightly lit)
const FIELD_EDGE = [11, 16, 23];   // #0b1017 — app theme colour
const EDGE_LINE = [34, 48, 71];    // #223047 — subtle inner border
const SKY = [56, 189, 248];        // #38bdf8 — shield outline / lower chevron
const SKY_BRIGHT = [125, 211, 252];// #7dd3fc — upper chevron
const HILITE = [191, 232, 253];    // #bfe8fd — apex catch-light dot
const INTERIOR = [14, 26, 43];     // #0e1a2b — slightly-lifted shield interior
const WHITE = [255, 255, 255];

// ---------- shield geometry ----------

// Reference geometry lives in a 170x170 box (shield vertically centred). Sample
// the shield's bezier outline into a polygon, then map every point into the icon
// frame — re-centring the shield's vertical bounding box on the frame centre and
// scaling about that centre by contentScale so every variant keeps a consistent,
// centred composition inside its own safe zone.
const REF = 170;
// Shield vertical extent is y=32..143 -> bbox centre 87.5. Shift so it lands on
// the frame centre (0.5).
const RECENTER_Y = 87.5 / REF - 0.5;

function cubic(p0, p1, p2, p3, n, out, includeEnd) {
  for (let i = 1; i <= (includeEnd ? n : n - 1); i++) {
    const t = i / n, mt = 1 - t;
    const a = mt * mt * mt, b = 3 * mt * mt * t, cc = 3 * mt * t * t, d = t * t * t;
    out.push([
      a * p0[0] + b * p1[0] + cc * p2[0] + d * p3[0],
      a * p0[1] + b * p1[1] + cc * p2[1] + d * p3[1],
    ]);
  }
}

// The shield outline as a dense polygon in reference (170-box) coords, matching
// path "M85 32 C104 37 122 42 126 48 L126 88 C126 114 108 132 85 143
//       C62 132 44 114 44 88 L44 48 C48 42 66 37 85 32 Z".
function sampleShieldRef() {
  const N = 18;
  const p = [[85, 32]];
  cubic([85, 32], [104, 37], [122, 42], [126, 48], N, p, true);
  p.push([126, 88]);
  cubic([126, 88], [126, 114], [108, 132], [85, 143], N, p, true);
  cubic([85, 143], [62, 132], [44, 114], [44, 88], N, p, true);
  p.push([44, 48]);
  // back up to the start — exclude the final point (equals the first vertex).
  cubic([44, 48], [48, 42], [66, 37], [85, 32], N, p, false);
  return p;
}

// Build the coordinate mapping + shield SDF for a given output size / scale.
function buildGeom(S, cs) {
  const X = (px) => S * (0.5 + (px / REF - 0.5) * cs);
  const Y = (py) => S * (0.5 + (py / REF - RECENTER_Y - 0.5) * cs);
  const L = (len) => (len / REF) * S * cs;         // reference length -> px
  const P = (px, py) => [X(px), Y(py)];
  const shieldPts = sampleShieldRef().map(([px, py]) => [X(px), Y(py)]);
  const cornerR = L(3);                            // soften the shoulder corners
  const sdShield = (x, y) => sdPolygon(x, y, shieldPts) - cornerR;
  return { X, Y, L, P, sdShield };
}

// Coverage for a chevron: two round-capped segments (^) meeting at an apex.
function chevronCoverage(g, a, apex, b, halfW) {
  const A = g.P(...a), M = g.P(...apex), B = g.P(...b);
  return (x, y) => {
    const d = Math.min(
      sdSegment(x, y, A[0], A[1], M[0], M[1]),
      sdSegment(x, y, M[0], M[1], B[0], B[1]),
    );
    return aa(d - halfW);
  };
}

// ---------- glyph composition ----------

/**
 * @param {number} size    output pixels
 * @param {object} opts
 *   fullBleed  — fill the whole frame (iOS / maskable); else a rounded badge
 *   contentScale — scale the shield about the centre (safe-zone control)
 *   rounded    — corner radius as a fraction of size (badge variants)
 *   opaque     — encode as RGB (no alpha); implies fullBleed
 */
function renderIcon(size, {
  fullBleed = false, contentScale = 0.92, rounded = 0.22, opaque = false,
} = {}) {
  const bleed = fullBleed || opaque;
  const c = new Canvas(size);
  const S = size;
  const cx = S / 2;
  const g = buildGeom(S, contentScale);

  // --- field: radial from a faintly-lit centre out to the theme colour ---
  const fieldColor = (x, y) => {
    const d = Math.hypot(x - cx, y - cx) / (S * 0.62);
    return mix(FIELD_CTR, FIELD_EDGE, clamp01(d));
  };
  if (bleed) {
    c.paintFn(fieldColor, () => 1);
  } else {
    const rr = S * rounded;
    c.paintFn(fieldColor, (x, y) => aa(sdRoundRect(x, y, cx, cx, S / 2, S / 2, rr)));
    // subtle inner border for depth
    const inset = Math.max(1.5, S * 0.012);
    c.paint(EDGE_LINE, (x, y) => {
      const o = sdRoundRect(x, y, cx, cx, S / 2 - inset, S / 2 - inset, rr * 0.92);
      return aa(Math.abs(o) - inset * 0.6);
    });
  }

  // --- shield: lifted dark interior + sky-blue outline ---
  c.paint(INTERIOR, (x, y) => aa(g.sdShield(x, y)));
  const shieldStroke = g.L(5) / 2;
  c.paint(SKY, (x, y) => aa(Math.abs(g.sdShield(x, y)) - shieldStroke));

  // --- ascending signal: lower chevron (dimmer), then upper (brighter) ---
  const lower = chevronCoverage(g, [70, 114], [85, 88], [100, 114], g.L(7) / 2);
  c.paint(SKY, (x, y) => lower(x, y) * 0.75);
  const upper = chevronCoverage(g, [64, 92], [85, 62], [106, 92], g.L(9) / 2);
  c.paint(SKY_BRIGHT, upper);

  // --- bright catch-light dot just above the top chevron ---
  const dot = g.P(85, 57);
  const dotR = g.L(4.5);
  c.paint(HILITE, (x, y) => aa(sdCircle(x, y, dot[0], dot[1], dotR)));

  return encodePng(size, c.px, { opaque });
}

// Monochrome badge (Android notification badge): white shield outline + white
// ascending chevron on transparent.
function renderBadge(size) {
  const c = new Canvas(size);
  const g = buildGeom(size, 0.88);
  const shieldStroke = g.L(6) / 2;
  c.paint(WHITE, (x, y) => aa(Math.abs(g.sdShield(x, y)) - shieldStroke));
  const chev = chevronCoverage(g, [64, 92], [85, 62], [106, 92], g.L(9) / 2);
  c.paint(WHITE, chev);
  const dot = g.P(85, 57);
  c.paint(WHITE, (x, y) => aa(sdCircle(x, y, dot[0], dot[1], g.L(4.5))));
  return encodePng(size, c.px);
}

// ---------- emit ----------

mkdirSync(OUT_DIR, { recursive: true });
const webFiles = {
  'icon-192.png': renderIcon(192, { contentScale: 1.08 }),
  'icon-512.png': renderIcon(512, { contentScale: 1.08 }),
  'maskable-192.png': renderIcon(192, { fullBleed: true, contentScale: 0.82 }),
  'maskable-512.png': renderIcon(512, { fullBleed: true, contentScale: 0.82 }),
  'apple-touch-icon.png': renderIcon(180, { fullBleed: true, contentScale: 0.96 }),
  'badge-96.png': renderBadge(96),
};
for (const [name, buf] of Object.entries(webFiles)) {
  writeFileSync(join(OUT_DIR, name), buf);
  console.log(`wrote public/icons/${name} (${buf.length} bytes)`);
}

// iOS master — opaque RGB 1024 (no alpha). Only when the appiconset exists.
if (existsSync(IOS_DIR)) {
  const ios = renderIcon(1024, { opaque: true, contentScale: 0.98 });
  const p = join(IOS_DIR, 'AppIcon-1024.png');
  writeFileSync(p, ios);
  console.log(`wrote ${p} (${ios.length} bytes, opaque RGB)`);
} else {
  console.log(`skipped iOS icon — ${IOS_DIR} not found`);
}
