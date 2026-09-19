/**
 * Copyright (c) 2026 The xterm.js authors. All rights reserved.
 * @license MIT
 */

import { IThemeService } from 'browser/services/Services';
import { Attributes, FgFlags } from 'common/buffer/Constants';

/**
 * The packed rgba a cell's background paints with, resolved the same way
 * for whoever draws it.
 *
 * Two renderers need this answer and must not disagree about it. The
 * rectangle renderer draws most backgrounds, while the glyph renderer
 * draws pixel-plane cells opaque with their background baked in, since
 * those cells move with the per-cell modes and a rectangle left behind
 * would not follow them.
 */
export function backgroundRgba(themeService: IThemeService, fg: number, bg: number): number {
  if (fg & FgFlags.INVERSE) {
    switch (fg & Attributes.CM_MASK) {
      case Attributes.CM_P16:
      case Attributes.CM_P256:
        return themeService.colors.ansi[fg & Attributes.PCOLOR_MASK].rgba;
      case Attributes.CM_RGB:
        return (fg & Attributes.RGB_MASK) << 8;
      case Attributes.CM_DEFAULT:
      default:
        return themeService.colors.foreground.rgba;
    }
  }
  switch (bg & Attributes.CM_MASK) {
    case Attributes.CM_P16:
    case Attributes.CM_P256:
      return themeService.colors.ansi[bg & Attributes.PCOLOR_MASK].rgba;
    case Attributes.CM_RGB:
      return (bg & Attributes.RGB_MASK) << 8;
    case Attributes.CM_DEFAULT:
    default:
      return themeService.colors.background.rgba;
  }
}
