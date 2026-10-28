/**
 * Copyright (c) 2026 The xterm.js authors. All rights reserved.
 * @license MIT
 */

import { assert } from 'chai';
import { BlendMode, gpuMode, isPlain, planeAlpha, planeRuns } from './PlaneLook';

describe('plane looks', () => {
  it('draws overlay and difference as normal on the GPU', () => {
    assert.equal(gpuMode(2), BlendMode.SCREEN);
    assert.equal(gpuMode(5.2), BlendMode.DARKEN);
    assert.equal(gpuMode(6), BlendMode.NORMAL);
    assert.equal(gpuMode(7), BlendMode.NORMAL);
    assert.isTrue(isPlain({ fade: 0, blend: 7 }));
    assert.isFalse(isPlain({ fade: 0.5, blend: 0 }));
    assert.equal(planeAlpha({ fade: 0.25, blend: 0 }), 0.75);
    assert.equal(planeAlpha(undefined), 1);
  });

  it('gives a plane with a look its own run and joins the plain ones', () => {
    const counts = new Int32Array(17);
    counts[15] = 4; counts[9] = 2; counts[3] = 6; counts[2] = 1; counts[1] = 5;
    const looks = { 3: { fade: 0, blend: 2 } };
    assert.deepEqual(planeRuns(counts, 15, looks), [[15, 9], [3, 3], [2, 1]]);
    assert.deepEqual(planeRuns(counts, 15, undefined), [[15, 1]]);
    assert.deepEqual(planeRuns(counts, 15, { 15: { fade: 0.5, blend: 0 }, 9: { fade: 0.5, blend: 0 } }), [[15, 15], [9, 9], [3, 1]]);
  });
});
