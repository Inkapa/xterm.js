/**
 * Copyright (c) 2026 The xterm.js authors. All rights reserved.
 * @license MIT
 */
import { assert } from 'chai';
import { Attributes, BgFlags, FgFlags } from 'common/buffer/Constants';
import { bgWithoutTag, fgWithoutTag, layerTag } from './LayerTag';

// The flags the door sets for each tag, as the parser leaves them in a cell.
function flagsFor(tag: number): { fg: number, bg: number } {
  let fg = 0;
  let bg = 0;
  if (tag & 1) { fg |= FgFlags.BLINK; }
  if (tag & 2) { fg |= FgFlags.STRIKETHROUGH; }
  if (tag & 4) { bg |= BgFlags.OVERLINE; }
  return { fg: fg >>> 0, bg: bg >>> 0 };
}

describe('layer tag', () => {
  it('reads every tag the door can send', () => {
    for (let tag = 0; tag <= 7; tag++) {
      const { fg, bg } = flagsFor(tag);
      assert.equal(layerTag(fg, bg), tag);
    }
  });

  it('is untagged for a cell with no flags', () => {
    assert.equal(layerTag(0, 0), 0);
    assert.equal(layerTag(Attributes.CM_RGB | 0x123456, Attributes.CM_RGB | 0x654321), 0);
  });

  it('removes the tag and leaves colour and other flags alone', () => {
    const colour = Attributes.CM_RGB | 0x336699;
    const others = FgFlags.BOLD | FgFlags.UNDERLINE | FgFlags.INVERSE;
    const otherBg = BgFlags.ITALIC | BgFlags.DIM;
    for (let tag = 0; tag <= 7; tag++) {
      const { fg, bg } = flagsFor(tag);
      const cleanFg = fgWithoutTag((colour | others | fg) >>> 0);
      const cleanBg = bgWithoutTag((colour | otherBg | bg) >>> 0);
      assert.equal(cleanFg, (colour | others) >>> 0, `tag ${tag} fg`);
      assert.equal(cleanBg, (colour | otherBg) >>> 0, `tag ${tag} bg`);
      assert.equal(layerTag(cleanFg, cleanBg), 0, `tag ${tag} is gone`);
    }
  });

  it('keeps words unsigned, so they compare equal to the render model', () => {
    const { fg } = flagsFor(2);
    assert.isAtLeast(fgWithoutTag((Attributes.CM_RGB | fg) >>> 0), 0);
  });
});
