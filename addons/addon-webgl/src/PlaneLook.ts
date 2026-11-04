/**
 * Copyright (c) 2026 The xterm.js authors. All rights reserved.
 * @license MIT
 */

import { IWebGL2RenderingContext } from './Types';

/**
 * How a plane goes over the planes beneath it: the page reads it from the
 * door's `fade` and `blend` effects on the plane's layer and sets it on
 * `window.glyph.planeLooks`, keyed by layer tag. The modes are glyph's
 * (`renderers/percell/src/blend.rs`); fade 0 shows the plane fully.
 */
export interface IPlaneLook {
  fade: number;
  blend: number;
}

export const enum BlendMode {
  NORMAL = 0,
  MULTIPLY = 1,
  SCREEN = 2,
  ADD = 3,
  LIGHTEN = 4,
  DARKEN = 5
}

/**
 * The mode a plane draws with on the GPU. Overlay (6) and difference (7)
 * have no blend equation and draw as normal.
 */
export function gpuMode(blend: number): BlendMode {
  const mode = Math.round(blend);
  return mode >= BlendMode.MULTIPLY && mode <= BlendMode.DARKEN ? mode : BlendMode.NORMAL;
}

/** Whether a plane draws as it would with no look at all. */
export function isPlain(look: IPlaneLook | undefined): boolean {
  return !look || (!(look.fade > 0) && gpuMode(look.blend) === BlendMode.NORMAL);
}

/** The share of the plane drawn, 1 for a plane with no fade. */
export function planeAlpha(look: IPlaneLook | undefined): number {
  return look && look.fade > 0 ? Math.max(0, 1 - look.fade) : 1;
}

/**
 * Sets the blend state for a mode. The glyph shader shapes its output to
 * match (see `u_planeBlend`): multiply and darken get the colour lightened
 * toward white by its alpha, screen, add and lighten the colour scaled by
 * it, so a glyph's edges and a faded plane blend in proportion.
 */
export function setBlend(gl: IWebGL2RenderingContext, mode: BlendMode): void {
  switch (mode) {
    case BlendMode.MULTIPLY:
      gl.blendEquation(gl.FUNC_ADD);
      gl.blendFunc(gl.DST_COLOR, gl.ZERO);
      break;
    case BlendMode.SCREEN:
      gl.blendEquation(gl.FUNC_ADD);
      gl.blendFunc(gl.ONE_MINUS_DST_COLOR, gl.ONE);
      break;
    case BlendMode.ADD:
      gl.blendEquation(gl.FUNC_ADD);
      gl.blendFunc(gl.ONE, gl.ONE);
      break;
    case BlendMode.LIGHTEN:
      gl.blendEquation(gl.MAX);
      break;
    case BlendMode.DARKEN:
      gl.blendEquation(gl.MIN);
      break;
    default:
      gl.blendEquation(gl.FUNC_ADD);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  }
}

/**
 * Splits the planes, drawn from the highest tag to the lowest, into runs
 * that share a draw: each plane with a look is a run of its own, and
 * neighbouring plain planes share one. `counts[tag]` is how many cells the
 * plane has. Returns [first tag, last tag] pairs in draw order.
 */
export function planeRuns(counts: ArrayLike<number>, maxTag: number, looks: Record<number, IPlaneLook> | undefined): [number, number][] {
  const runs: [number, number][] = [];
  for (let tag = maxTag; tag >= 1; tag--) {
    if (counts[tag] === 0) {
      continue;
    }
    const last = runs[runs.length - 1];
    if (last && isPlain(looks?.[tag]) && isPlain(looks?.[last[0]])) {
      last[1] = tag;
    } else {
      runs.push([tag, tag]);
    }
  }
  return runs;
}
