/**
 * Copyright (c) 2018 The xterm.js authors. All rights reserved.
 * @license MIT
 */

import { throwIfFalsy } from 'browser/renderer/shared/RendererUtils';
import { Disposable, toDisposable } from 'common/Lifecycle';
import { IWebGL2RenderingContext, IWebGLVertexArrayObject } from './Types';
import { createProgram } from './WebglUtils';
import { intensityValue } from './GlyphRenderer';

// The glyph control surface: window.glyph, created by the page.
function glyphCfg(): any {
  return (typeof globalThis !== 'undefined' && (globalThis as any).glyph) || {};
}

// moshModeValue maps the page's mosh field onto the u_mode value. Unlike the
// fragment filters this stage does not stack: the modes differ only in how the
// history is displaced, and two displacements at once is just a third one.
function moshModeValue(name: string | undefined): number {
  switch ((name || '').trim()) {
    case 'trail': return 1;
    case 'smear': return 2;
    case 'creep': return 3;
    case 'swell': return 4;
    case 'suck': return 5;
    default: return 0;
  }
}

// How much of the previous composite survives into the next frame, before the
// master intensity and the mosh knob scale it. The frame is added on top
// rather than crossfaded, so the trails keep their brightness and the live
// text stays legible; the price is that the sum saturates, which is what caps
// this at well under 1.
const DECAY = 0.6;
const DECAY_MAX = 0.92;

// The fullscreen quad is generated from gl_VertexID, so the pass owns no
// vertex buffer: an empty vertex array object is enough to keep the other
// renderers' attribute state out of the draw.
const vertexShaderSource = `#version 300 es
out vec2 v_uv;

void main() {
  v_uv = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1));
  gl_Position = vec4(v_uv * 2.0 - 1.0, 0.0, 1.0);
}`;

const fragmentShaderSource = `#version 300 es
precision lowp float;

in vec2 v_uv;

uniform sampler2D u_history;
// u_mode picks how the history is displaced before it is drawn back.
uniform int u_mode;

out vec4 outColor;

void main() {
  // The history is redrawn one step off its own position, so each frame's
  // ghost lands slightly further along and the trail streaks instead of
  // simply dimming in place. 'trail' leaves it in place for a plain decay.
  vec2 uv = v_uv;
  if (u_mode == 2) {
    uv.x -= 0.0025;
  } else if (u_mode == 3) {
    uv.y -= 0.0025;
  } else if (u_mode == 4) {
    uv = (uv - 0.5) * 0.996 + 0.5;
  } else if (u_mode == 5) {
    uv = (uv - 0.5) * 1.004 + 0.5;
  }
  outColor = texture(u_history, uv);
}`;

/**
 * The frame-feedback (datamosh) pass: the finished frame is kept in a texture
 * and drawn back over the next one at a decay, so bright cells smear into
 * trails that fade on their own.
 *
 * It runs after the background, glyph and cursor passes, on the default
 * framebuffer rather than an offscreen target: the history texture is filled
 * with `copyTexSubImage2D` straight off the drawing buffer, which costs one
 * full-frame copy and saves the whole ping-pong FBO apparatus. The trails
 * therefore advance once per rendered frame, not on a clock: an idle terminal
 * freezes them until the next write.
 */
export class MoshRenderer extends Disposable {
  private _program: WebGLProgram;
  private _vertexArrayObject: IWebGLVertexArrayObject;
  private _texture: WebGLTexture;
  private _historyLocation: WebGLUniformLocation;
  private _modeLocation: WebGLUniformLocation;
  // The atlas claims texture units 0..maxAtlasPages-1 and only rebinds a page
  // when its version changes, so a unit borrowed from that range would leave a
  // stale binding behind. Take the last unit instead, which it never reaches.
  private _unit: number;
  private _width: number = 0;
  private _height: number = 0;
  // Whether the texture holds a frame yet. False after a resize or while the
  // pass is off, so the first frame back seeds the history instead of blending
  // against whatever was left in it.
  private _primed: boolean = false;

  constructor(
    private _gl: IWebGL2RenderingContext
  ) {
    super();

    const gl = this._gl;

    this._program = throwIfFalsy(createProgram(gl, vertexShaderSource, fragmentShaderSource));
    this.register(toDisposable(() => gl.deleteProgram(this._program)));

    this._historyLocation = throwIfFalsy(gl.getUniformLocation(this._program, 'u_history'));
    this._modeLocation = throwIfFalsy(gl.getUniformLocation(this._program, 'u_mode'));

    this._vertexArrayObject = throwIfFalsy(gl.createVertexArray());

    this._unit = gl.getParameter(gl.MAX_COMBINED_TEXTURE_IMAGE_UNITS) - 1;
    this._texture = throwIfFalsy(gl.createTexture());
    this.register(toDisposable(() => gl.deleteTexture(this._texture)));
    gl.activeTexture(gl.TEXTURE0 + this._unit);
    gl.bindTexture(gl.TEXTURE_2D, this._texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  }

  public render(): void {
    const gl = this._gl;
    const mode = moshModeValue(glyphCfg().mosh);
    if (mode === 0) {
      // Drop the stale frame so toggling the pass back on does not pop a
      // ghost of whatever was on screen when it was switched off.
      this._primed = false;
      return;
    }

    const width = gl.canvas.width;
    const height = gl.canvas.height;
    if (width === 0 || height === 0) {
      return;
    }

    gl.activeTexture(gl.TEXTURE0 + this._unit);
    gl.bindTexture(gl.TEXTURE_2D, this._texture);
    if (width !== this._width || height !== this._height) {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      this._width = width;
      this._height = height;
      this._primed = false;
    }

    if (this._primed) {
      gl.useProgram(this._program);
      gl.bindVertexArray(this._vertexArrayObject);
      gl.uniform1i(this._historyLocation, this._unit);
      gl.uniform1i(this._modeLocation, mode);
      // dst = history * decay + frame. The decay rides in as the blend
      // constant, which keeps the whole feedback loop to one draw: no second
      // framebuffer to read the destination from.
      const decay = Math.min(DECAY * decayScale() * intensityValue(), DECAY_MAX);
      gl.blendColor(decay, decay, decay, decay);
      gl.blendFunc(gl.CONSTANT_COLOR, gl.ONE);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      // The other renderers set the blend mode once at startup and never
      // touch it again, so this pass has to put it back.
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    }

    gl.copyTexSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 0, 0, width, height);
    this._primed = true;
  }
}

// decayScale reads the feedback multiplier from the page's params bag, on the
// same footing as the u_param[] slots: 1.0 is the built-in look, higher values
// hold the trails longer. It is a params entry rather than a u_param slot
// because the decay never reaches the shader, it is the blend constant.
function decayScale(): number {
  const p = glyphCfg().params || {};
  const v = p.mosh;
  return typeof v === 'number' ? Math.max(v, 0) : 1.0;
}
