/**
 * Copyright (c) 2026 The xterm.js authors. All rights reserved.
 * @license MIT
 */
import { assert } from 'chai';
import { Attributes } from 'common/buffer/Constants';
import { applyPayload, cubeRgb, hashLine, PlaneRow } from './PlanePayload';

const b64 = (bytes: number[]): string => btoa(String.fromCharCode(...bytes));
const rgb = (r: number, g: number, b: number): number => (Attributes.CM_RGB | (r << 16) | (g << 8) | b) >>> 0;

describe('plane payload', () => {
  it('reads a run of opaque cells', () => {
    const row = new PlaneRow();
    // start 4, one run: length 2, flags 0, glyph '.', fg 1 2 3, bg 4 5 6
    assert.isTrue(applyPayload(row, 3, b64([4, 1, 2, 0, 0x2e, 1, 2, 3, 4, 5, 6]), 20));
    assert.equal(row.count, 2);
    assert.deepEqual(Array.from(row.x.subarray(0, 2)), [4, 5]);
    assert.equal(row.tag[0], 3);
    assert.equal(row.code[0], 0x2e);
    assert.equal(row.fg[0], rgb(1, 2, 3));
    assert.equal(row.bg[1], rgb(4, 5, 6));
    assert.equal(row.over[0], 0);
  });

  it('reads a glyph over a transparent background and a marker', () => {
    const row = new PlaneRow();
    // segment at 8: an over cell 'o' (no bg bytes), then a marker at 9
    assert.isTrue(applyPayload(row, 1, b64([8, 2, 1, 1, 0x6f, 7, 7, 7, 1, 2]), 20));
    assert.equal(row.count, 1);
    assert.equal(row.over[0], 1);
    assert.equal(row.bg[0], 0);
    assert.deepEqual(row.markers, [9]);
  });

  it('reads multi-byte lengths and glyphs', () => {
    const row = new PlaneRow();
    // length 300 (0xac 0x02), glyph U+2580 (0x80 0x4b)
    assert.isTrue(applyPayload(row, 2, b64([0, 1, 0xac, 0x02, 0, 0x80, 0x4b, 9, 9, 9, 1, 1, 1]), 400));
    assert.equal(row.count, 300);
    assert.equal(row.code[299], 0x2580);
    assert.equal(row.x[299], 299);
  });

  it('reads cube palette indices', () => {
    const row = new PlaneRow();
    assert.isTrue(applyPayload(row, 3, b64([0, 1, 1, 4, 0x61, 196, 21]), 20));
    assert.equal(row.fg[0], rgb(255, 0, 0));
    assert.equal(row.bg[0], rgb(0, 0, 255));
    assert.equal(cubeRgb(232), 0x080808);
  });

  it('keeps cells inside the row', () => {
    const row = new PlaneRow();
    assert.isTrue(applyPayload(row, 3, b64([2, 1, 10, 0, 0x2e, 1, 1, 1, 2, 2, 2]), 5));
    assert.equal(row.count, 3);
  });

  it('drops a truncated or corrupt payload whole', () => {
    const row = new PlaneRow();
    assert.isFalse(applyPayload(row, 3, b64([4, 1, 2, 0, 0x2e, 1, 2, 3]), 20));
    assert.equal(row.count, 0);
    assert.isFalse(applyPayload(row, 3, '!!!not base64', 20));
    assert.equal(row.count, 0);
  });

  it('hashes a row by its cells', () => {
    const cells = [[1, 2, 3], [4, 5, 6]];
    const line = (data: number[][]) => ({
      getFg: (x: number) => data[x][0],
      getBg: (x: number) => data[x][1],
      getCodePoint: (x: number) => data[x][2]
    });
    const before = hashLine(line(cells), 2);
    assert.equal(before, hashLine(line(cells), 2));
    cells[1][2] = 7;
    assert.notEqual(before, hashLine(line(cells), 2));
  });
});
