/**
 * Copyright (c) 2026 The xterm.js authors. All rights reserved.
 * @license MIT
 */

import { BgFlags, FgFlags } from 'common/buffer/Constants';

/**
 * The layer a scene assigned to a cell. The page's door sends it as SGR
 * attribute flags that nothing else in the stream uses: blink is bit 0,
 * strikethrough bit 1, overline bit 2 and italic bit 3, which the door states
 * only once a layer past seven is in use. The value is 0 when the scene did
 * not say, else the layer's index plus one.
 */
export function layerTag(fg: number, bg: number): number {
  return ((fg & FgFlags.BLINK) ? 1 : 0) | ((fg & FgFlags.STRIKETHROUGH) ? 2 : 0) | ((bg & BgFlags.OVERLINE) ? 4 : 0) | ((bg & BgFlags.ITALIC) ? 8 : 0);
}

/** The largest tag: four bits, zero meaning unassigned. */
export const MAX_LAYER_TAG = 15;

const FG_TAG_BITS = FgFlags.BLINK | FgFlags.STRIKETHROUGH;

/**
 * The foreground word without the tag's bits. Strikethrough and overline
 * would otherwise draw a line through the cell and split its atlas entry, so
 * the tag is read and then removed before anything else sees the cell.
 */
export function fgWithoutTag(fg: number): number {
  return (fg & ~FG_TAG_BITS) >>> 0;
}

/**
 * The background word without the tag's bits. See fgWithoutTag; italic would
 * also slant the glyph.
 */
export function bgWithoutTag(bg: number): number {
  return (bg & ~(BgFlags.OVERLINE | BgFlags.ITALIC)) >>> 0;
}
