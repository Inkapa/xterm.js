/**
 * Copyright (c) 2026 The xterm.js authors. All rights reserved.
 * @license MIT
 */

import { Attributes } from 'common/buffer/Constants';

/**
 * Plane payloads. The door writes, after a row's cells, `ESC ] 7770 ; tag ;
 * base64 BEL` with the cells of one plane that another plane covers, and
 * `ESC ] 7771 ; shift BEL` at the start of every row of a frame with planes (glyph's
 * docs/renderers.md, "Layers"). The layout is written by
 * internal/engine/planepayload.go and read by renderers/native/src/plane.rs,
 * which this file must agree with byte for byte: a body is a list of segments
 * (start column, run count, runs), a run is (length, flags, glyph, colours),
 * and integers are unsigned LEB128.
 */

const FLAG_OVER = 1;
const FLAG_MARKER = 2;
const FLAG_CUBE = 4;

/**
 * The plane cells one row holds, in parallel arrays that are reused from
 * frame to frame so a payload allocates nothing once they have grown. `x` is
 * the column before the row's shift. Colours are xterm colour words (RGB
 * mode), the same form the renderer resolves a cell to.
 */
export class PlaneRow {
  public shift = 0;
  /** The row's flat cells when the payload arrived, see hashLine. */
  public hash = 0;
  public count = 0;
  public x = new Uint16Array(32);
  public tag = new Uint8Array(32);
  public code = new Uint32Array(32);
  public fg = new Uint32Array(32);
  public bg = new Uint32Array(32);
  /** 1 for a glyph over a transparent background. */
  public over = new Uint8Array(32);
  /** Columns (before the shift) whose top-plane cell has a transparent background. */
  public markers: number[] = [];

  public reset(): void {
    this.count = 0;
    this.shift = 0;
    this.markers.length = 0;
  }

  public grow(needed: number): void {
    if (needed <= this.x.length) {
      return;
    }
    let size = this.x.length;
    while (size < needed) {
      size *= 2;
    }
    const widen = <T extends Uint8Array | Uint16Array | Uint32Array>(old: T): T => {
      const wider = new (old.constructor as new (n: number) => T)(size);
      wider.set(old);
      return wider;
    };
    this.x = widen(this.x);
    this.tag = widen(this.tag);
    this.code = widen(this.code);
    this.fg = widen(this.fg);
    this.bg = widen(this.bg);
    this.over = widen(this.over);
  }
}

/**
 * The xterm-256 colour for a palette index: the 6x6x6 cube, then the gray
 * ramp. Indices below 16 are the terminal's theme colours, which the door never
 * sends, so they read as black.
 */
export function cubeRgb(index: number): number {
  if (index >= 16 && index <= 231) {
    const level = (n: number): number => n === 0 ? 0 : 55 + 40 * n;
    const i = index - 16;
    return (level(Math.floor(i / 36)) << 16) | (level(Math.floor(i / 6) % 6) << 8) | level(i % 6);
  }
  if (index >= 232) {
    const gray = 8 + 10 * (index - 232);
    return (gray << 16) | (gray << 8) | gray;
  }
  return 0;
}

const RGB_WORD = Attributes.CM_RGB;

/**
 * Reads one payload into `row`, appending its cells under `tag` (1 to 7, the
 * layer's index plus one). Returns false, leaving the row as it was, for
 * anything malformed: a payload is dropped whole rather than half applied.
 */
export function applyPayload(row: PlaneRow, tag: number, base64: string, cols: number): boolean {
  let raw: string;
  try {
    raw = atob(base64);
  } catch {
    return false;
  }
  const start = row.count;
  const markerStart = row.markers.length;
  let at = 0;
  const end = raw.length;
  let bad = false;
  const byte = (): number => {
    if (at >= end) {
      bad = true;
      return 0;
    }
    return raw.charCodeAt(at++);
  };
  const varint = (): number => {
    let value = 0;
    for (let shift = 0; shift < 35; shift += 7) {
      const b = byte();
      value += (b & 0x7f) * Math.pow(2, shift);
      if (b < 0x80) {
        return value;
      }
    }
    bad = true;
    return 0;
  };

  while (at < end && !bad) {
    let column = varint();
    const runs = varint();
    for (let r = 0; r < runs && !bad; r++) {
      const length = varint();
      const flags = byte();
      if (flags & FLAG_MARKER) {
        for (let k = 0; k < length; k++) {
          row.markers.push(column + k);
        }
        column += length;
        continue;
      }
      const code = varint();
      const cube = (flags & FLAG_CUBE) !== 0;
      const over = (flags & FLAG_OVER) !== 0;
      const fg = cube ? cubeRgb(byte()) : ((byte() << 16) | (byte() << 8) | byte());
      const bg = over ? 0 : (cube ? cubeRgb(byte()) : ((byte() << 16) | (byte() << 8) | byte()));
      if (bad) {
        break;
      }
      // Cells shifted or written past the last column cannot show, and a
      // length is not trusted beyond the row.
      const visible = Math.min(length, Math.max(0, cols - column));
      row.grow(row.count + visible);
      for (let k = 0; k < visible; k++) {
        const n = row.count++;
        row.x[n] = column + k;
        row.tag[n] = tag;
        row.code[n] = code;
        row.fg[n] = (RGB_WORD | fg) >>> 0;
        row.bg[n] = over ? 0 : (RGB_WORD | bg) >>> 0;
        row.over[n] = over ? 1 : 0;
      }
      column += length;
    }
  }
  if (bad) {
    row.count = start;
    row.markers.length = markerStart;
    return false;
  }
  return true;
}

/** The parts of a buffer line the row hash reads. */
export interface IHashableLine {
  getFg(x: number): number;
  getBg(x: number): number;
  getCodePoint(x: number): number;
}

/**
 * A hash of a row's flat cells. A payload is only good for the paint it came
 * with, and the buffer does not say when a row was painted over, so a payload
 * remembers what its row held and is ignored once the row holds something else.
 */
export function hashLine(line: IHashableLine, cols: number): number {
  let hash = 0x811c9dc5;
  for (let x = 0; x < cols; x++) {
    hash = Math.imul(hash ^ line.getFg(x), 0x01000193);
    hash = Math.imul(hash ^ line.getBg(x), 0x01000193);
    hash = Math.imul(hash ^ line.getCodePoint(x), 0x01000193);
  }
  return hash >>> 0;
}
