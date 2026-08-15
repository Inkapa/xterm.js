/**
 * Copyright (c) 2018 The xterm.js authors. All rights reserved.
 * @license MIT
 */

// shaderBgBit maps one background filter name onto its bit in the u_badgl
// mask, in the order the filters compose in the shader.
function shaderBgBit(name: string): number {
  switch (name.trim()) {
    case 'static': return 1;
    case 'bands': return 2;
    case 'scan': return 4;
    case 'chan': return 8;
    case 'neg': return 16;
    case 'fade': return 32;
    case 'flicker': return 64;
    case 'vignette': return 128;
    case 'mosaic': return 256;
    case 'sweep': return 512;
    case 'dither': return 1024;
    case 'bayer': return 2048;
    default: return 0;
  }
}

// shaderBgModeValue turns the page's background shader field into the
// u_badgl bitmask, accepting a comma-separated list so filters stack.
function shaderBgModeValue(name: string | undefined): number {
  if (!name) {
    return 0;
  }
  let mask = 0;
  for (const part of name.split(',')) {
    mask |= shaderBgBit(part);
  }
  return mask;
}

// The glyph control surface: window.glyph, created by the page.
function glyphCfg(): any {
  return (typeof globalThis !== 'undefined' && (globalThis as any).glyph) || {};
}

import { throwIfFalsy } from 'browser/renderer/shared/RendererUtils';
import { IRenderDimensions } from 'browser/renderer/shared/Types';
import { IThemeService } from 'browser/services/Services';
import { ReadonlyColorSet } from 'browser/Types';
import { Attributes, FgFlags } from 'common/buffer/Constants';
import { Disposable, toDisposable } from 'common/Lifecycle';
import { IColor } from 'common/Types';
import { Terminal } from '@xterm/xterm';
import { RENDER_MODEL_BG_OFFSET, RENDER_MODEL_FG_OFFSET, RENDER_MODEL_INDICIES_PER_CELL } from './RenderModel';
import { IRenderModel, IWebGL2RenderingContext, IWebGLVertexArrayObject } from './Types';
import { createProgram, expandFloat32Array, PROJECTION_MATRIX } from './WebglUtils';
import { vtxModeValue } from './GlyphRenderer';

const enum VertexAttribLocations {
  POSITION = 0,
  SIZE = 1,
  COLOR = 2,
  UNIT_QUAD = 3
}

const vertexShaderSource = `#version 300 es
layout (location = ${VertexAttribLocations.POSITION}) in vec2 a_position;
layout (location = ${VertexAttribLocations.SIZE}) in vec2 a_size;
layout (location = ${VertexAttribLocations.COLOR}) in vec4 a_color;
layout (location = ${VertexAttribLocations.UNIT_QUAD}) in vec2 a_unitquad;

uniform mat4 u_projection;
uniform highp int u_vtx;
uniform highp float u_vtime;

out vec4 v_color;

// warpVertex must match the glyph renderer's block exactly so the
// background bends together with the foreground (window.glyph.vtx).
vec4 warpVertex(vec4 pos) {
  vec2 p = pos.xy;
  if (u_vtx == 1) {
    pos.xy = p * (1.0 + 0.18 * dot(p, p));
  } else if (u_vtx == 2) {
    pos.x += p.y * 0.25;
  } else if (u_vtx == 3) {
    pos.xy += vec2(sin(u_vtime * 1.7), cos(u_vtime * 2.3)) * 0.02;
  } else if (u_vtx == 4) {
    pos.x += sin(p.y * 10.0 + u_vtime * 0.2) * 0.03;
  } else if (u_vtx == 5) {
    pos.xy = p * (1.0 - 0.25 * exp(-dot(p, p) * 3.0));
  } else if (u_vtx == 6) {
    float a = length(p) * 0.8 + u_vtime * 0.01;
    float s = sin(a), c = cos(a);
    pos.xy = mat2(c, -s, s, c) * p;
  } else if (u_vtx == 7) {
    pos.y += mod(u_vtime * 0.02, 2.0) - 1.0;
  }
  return pos;
}

void main() {
  vec2 zeroToOne = a_position + (a_unitquad * a_size);
  gl_Position = warpVertex(u_projection * vec4(zeroToOne, 0.0, 1.0));
  v_color = a_color;
}`;

const fragmentShaderSource = `#version 300 es
precision lowp float;

in vec4 v_color;

uniform highp int u_badgl;
uniform highp float u_time;
uniform highp vec2 u_resolution;

out vec4 outColor;

// 4x4 Bayer ordered-dither threshold matrix (kept in sync with the glyph
// renderer). Ordered dithering is the single-pass, O(1)-per-fragment
// stand-in for error diffusion, which is serial and cannot run in one pass.
const float bayer4x4[16] = float[16](
   0.0/16.0,  8.0/16.0,  2.0/16.0, 10.0/16.0,
  12.0/16.0,  4.0/16.0, 14.0/16.0,  6.0/16.0,
   3.0/16.0, 11.0/16.0,  1.0/16.0,  9.0/16.0,
  15.0/16.0,  7.0/16.0, 13.0/16.0,  5.0/16.0);

float ditherThreshold(vec2 fragCoord) {
  ivec2 p = ivec2(mod(fragCoord, 4.0));
  return bayer4x4[p.y * 4 + p.x];
}

void main() {
  outColor = v_color;

  // bad-GL background post-processing (window.glyph.shaderBg). u_badgl is a
  // bitmask (one bit per filter), so several stack in one pass and compose
  // in ascending bit order; shaderBgModeValue ORs the requested modes.
  // Colour-only: the background pass has no texture to re-sample, so the
  // lines and chroma modes are not offered here.
  if ((u_badgl & 1) != 0) {
    // static: animated hash noise over the whole frame
    float n = fract(sin(dot(gl_FragCoord.xy + vec2(u_time * 91.7, u_time * 47.3), vec2(12.9898, 78.233))) * 43758.5453);
    outColor.rgb = mix(outColor.rgb, vec3(n), 0.35);
  }
  if ((u_badgl & 2) != 0) {
    // bands: darken rows in a moving band
    float h = fract(sin(gl_FragCoord.y * 12.9898 + u_time * 53.0) * 43758.5453);
    if (h > 0.7) {
      outColor.rgb *= 0.6;
    }
  }
  if ((u_badgl & 4) != 0) {
    // scan: darken alternating rows
    if (mod(floor(gl_FragCoord.y), 2.0) < 1.0) {
      outColor.rgb *= 0.6;
    }
  }
  if ((u_badgl & 8) != 0) {
    // chan: swap the colour channels
    outColor.rgb = outColor.gbr;
  }
  if ((u_badgl & 16) != 0) {
    // neg: invert
    outColor = vec4(1.0) - outColor;
  }
  if ((u_badgl & 32) != 0) {
    // fade: translucent background
    outColor.a *= 0.7;
  }
  if ((u_badgl & 64) != 0) {
    // flicker: brightness pulses softly
    outColor.rgb *= 0.92 + 0.08 * sin(u_time * 7.0 + gl_FragCoord.x * 0.13);
  }
  if ((u_badgl & 128) != 0) {
    // vignette: darken toward the screen edges
    vec2 uv = gl_FragCoord.xy / u_resolution;
    float d = distance(uv, vec2(0.5));
    outColor.rgb *= 1.0 - smoothstep(0.35, 0.78, d) * 0.6;
  }
  if ((u_badgl & 256) != 0) {
    // mosaic: block-quantised noise modulates the background colour
    vec2 blk = floor(gl_FragCoord.xy / 16.0);
    float n = fract(sin(dot(blk, vec2(12.9898, 78.233)) + u_time * 0.1) * 43758.5453);
    outColor.rgb *= 0.4 + 0.6 * n;
  }
  if ((u_badgl & 512) != 0) {
    // sweep: a bright bar scans down the screen
    float sy = gl_FragCoord.y / u_resolution.y;
    float bar = smoothstep(0.03, 0.0, abs(fract(sy - u_time * 0.01) - 0.5));
    outColor.rgb += bar * 0.5;
  }
  if ((u_badgl & 1024) != 0) {
    // dither: 1-bit ordered dithering of the background luminance, the
    // full-screen colour field broken into a black/white stipple
    float lum = dot(outColor.rgb, vec3(0.299, 0.587, 0.114));
    outColor.rgb = vec3(step(ditherThreshold(gl_FragCoord.xy), lum));
  }
  if ((u_badgl & 2048) != 0) {
    // bayer: colour ordered dither, 4 levels per channel, so the field
    // posterises into stippled bands instead of smooth gradients
    float d = ditherThreshold(gl_FragCoord.xy) - 0.5;
    outColor.rgb = clamp(floor(outColor.rgb * 3.0 + 0.5 + d), 0.0, 3.0) / 3.0;
  }
}`;;

const INDICES_PER_RECTANGLE = 8;
const BYTES_PER_RECTANGLE = INDICES_PER_RECTANGLE * Float32Array.BYTES_PER_ELEMENT;

const INITIAL_BUFFER_RECTANGLE_CAPACITY = 20 * INDICES_PER_RECTANGLE;

class Vertices {
  public attributes: Float32Array;
  public count: number;

  constructor() {
    this.attributes = new Float32Array(INITIAL_BUFFER_RECTANGLE_CAPACITY);
    this.count = 0;
  }
}

// Work variables to avoid garbage collection
let $rgba = 0;
let $x1 = 0;
let $y1 = 0;
let $r = 0;
let $g = 0;
let $b = 0;
let $a = 0;

export class RectangleRenderer extends Disposable {

  private _program: WebGLProgram;
  private _vertexArrayObject: IWebGLVertexArrayObject;
  private _attributesBuffer: WebGLBuffer;
  private _projectionLocation: WebGLUniformLocation;
  private _badglLocation: WebGLUniformLocation;
  private _timeLocation: WebGLUniformLocation;
  private _fragResolutionLocation: WebGLUniformLocation;
  private _vtxLocation: WebGLUniformLocation;
  private _vtimeLocation: WebGLUniformLocation;
  private _frameCount: number = 0;
  private _bgFloat!: Float32Array;
  private _cursorFloat!: Float32Array;

  private _vertices: Vertices = new Vertices();
  private _verticesCursor: Vertices = new Vertices();

  constructor(
    private _terminal: Terminal,
    private _gl: IWebGL2RenderingContext,
    private _dimensions: IRenderDimensions,
    private readonly _themeService: IThemeService
  ) {
    super();

    const gl = this._gl;

    this._program = throwIfFalsy(createProgram(gl, vertexShaderSource, fragmentShaderSource));
    this.register(toDisposable(() => gl.deleteProgram(this._program)));

    // Uniform locations
    this._projectionLocation = throwIfFalsy(gl.getUniformLocation(this._program, 'u_projection'));
    this._badglLocation = throwIfFalsy(gl.getUniformLocation(this._program, 'u_badgl'));
    this._timeLocation = throwIfFalsy(gl.getUniformLocation(this._program, 'u_time'));
    this._fragResolutionLocation = throwIfFalsy(gl.getUniformLocation(this._program, 'u_resolution'));
    this._vtxLocation = throwIfFalsy(gl.getUniformLocation(this._program, 'u_vtx'));
    this._vtimeLocation = throwIfFalsy(gl.getUniformLocation(this._program, 'u_vtime'));

    // Create and set the vertex array object
    this._vertexArrayObject = gl.createVertexArray();
    gl.bindVertexArray(this._vertexArrayObject);

    // Setup a_unitquad, this defines the 4 vertices of a rectangle
    const unitQuadVertices = new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]);
    const unitQuadVerticesBuffer = gl.createBuffer();
    this.register(toDisposable(() => gl.deleteBuffer(unitQuadVerticesBuffer)));
    gl.bindBuffer(gl.ARRAY_BUFFER, unitQuadVerticesBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, unitQuadVertices, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(VertexAttribLocations.UNIT_QUAD);
    gl.vertexAttribPointer(VertexAttribLocations.UNIT_QUAD, 2, this._gl.FLOAT, false, 0, 0);

    // Setup the unit quad element array buffer, this points to indices in
    // unitQuadVertices to allow is to draw 2 triangles from the vertices via a
    // triangle strip
    const unitQuadElementIndices = new Uint8Array([0, 1, 2, 3]);
    const elementIndicesBuffer = gl.createBuffer();
    this.register(toDisposable(() => gl.deleteBuffer(elementIndicesBuffer)));
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, elementIndicesBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, unitQuadElementIndices, gl.STATIC_DRAW);

    // Setup attributes
    this._attributesBuffer = throwIfFalsy(gl.createBuffer());
    this.register(toDisposable(() => gl.deleteBuffer(this._attributesBuffer)));
    gl.bindBuffer(gl.ARRAY_BUFFER, this._attributesBuffer);
    gl.enableVertexAttribArray(VertexAttribLocations.POSITION);
    gl.vertexAttribPointer(VertexAttribLocations.POSITION, 2, gl.FLOAT, false, BYTES_PER_RECTANGLE, 0);
    gl.vertexAttribDivisor(VertexAttribLocations.POSITION, 1);
    gl.enableVertexAttribArray(VertexAttribLocations.SIZE);
    gl.vertexAttribPointer(VertexAttribLocations.SIZE, 2, gl.FLOAT, false, BYTES_PER_RECTANGLE, 2 * Float32Array.BYTES_PER_ELEMENT);
    gl.vertexAttribDivisor(VertexAttribLocations.SIZE, 1);
    gl.enableVertexAttribArray(VertexAttribLocations.COLOR);
    gl.vertexAttribPointer(VertexAttribLocations.COLOR, 4, gl.FLOAT, false, BYTES_PER_RECTANGLE, 4 * Float32Array.BYTES_PER_ELEMENT);
    gl.vertexAttribDivisor(VertexAttribLocations.COLOR, 1);

    this._updateCachedColors(_themeService.colors);
    this.register(this._themeService.onChangeColors(e => {
      this._updateCachedColors(e);
      this._updateViewportRectangle();
    }));
  }

  public renderBackgrounds(): void {
    this._renderVertices(this._vertices);
  }

  public renderCursor(): void {
    this._renderVertices(this._verticesCursor);
  }

  private _renderVertices(vertices: Vertices): void {
    const gl = this._gl;

    gl.useProgram(this._program);

    gl.bindVertexArray(this._vertexArrayObject);

    // bad-GL background shader modes, read live per frame.
    gl.uniform1i(this._badglLocation, shaderBgModeValue(glyphCfg().shaderBg));
    gl.uniform1f(this._timeLocation, this._frameCount);
    gl.uniform1i(this._vtxLocation, vtxModeValue(glyphCfg().vtx));
    gl.uniform1f(this._vtimeLocation, this._frameCount);
    this._frameCount++;
    gl.uniform2f(this._fragResolutionLocation, gl.canvas.width, gl.canvas.height);

    gl.uniformMatrix4fv(this._projectionLocation, false, PROJECTION_MATRIX);

    // Bind attributes buffer and draw
    gl.bindBuffer(gl.ARRAY_BUFFER, this._attributesBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, vertices.attributes, gl.DYNAMIC_DRAW);
    gl.drawElementsInstanced(this._gl.TRIANGLE_STRIP, 4, gl.UNSIGNED_BYTE, 0, vertices.count);
  }

  public handleResize(): void {
    this._updateViewportRectangle();
  }

  public setDimensions(dimensions: IRenderDimensions): void {
    this._dimensions = dimensions;
  }

  private _updateCachedColors(colors: ReadonlyColorSet): void {
    this._bgFloat = this._colorToFloat32Array(colors.background);
    this._cursorFloat = this._colorToFloat32Array(colors.cursor);
  }

  private _updateViewportRectangle(): void {
    // Set first rectangle that clears the screen
    this._addRectangleFloat(
      this._vertices.attributes,
      0,
      0,
      0,
      this._terminal.cols * this._dimensions.device.cell.width,
      this._terminal.rows * this._dimensions.device.cell.height,
      this._bgFloat
    );
  }

  public updateBackgrounds(model: IRenderModel): void {
    const terminal = this._terminal;
    const vertices = this._vertices;

    // Declare variable ahead of time to avoid garbage collection
    let rectangleCount = 1;
    let y: number;
    let x: number;
    let currentStartX: number;
    let currentBg: number;
    let currentFg: number;
    let currentInverse: boolean;
    let modelIndex: number;
    let bg: number;
    let fg: number;
    let inverse: boolean;
    let offset: number;

    for (y = 0; y < terminal.rows; y++) {
      currentStartX = -1;
      currentBg = 0;
      currentFg = 0;
      currentInverse = false;
      for (x = 0; x < terminal.cols; x++) {
        modelIndex = ((y * terminal.cols) + x) * RENDER_MODEL_INDICIES_PER_CELL;
        bg = model.cells[modelIndex + RENDER_MODEL_BG_OFFSET];
        fg = model.cells[modelIndex + RENDER_MODEL_FG_OFFSET];
        inverse = !!(fg & FgFlags.INVERSE);
        if (bg !== currentBg || (fg !== currentFg && (currentInverse || inverse))) {
          // A rectangle needs to be drawn if going from non-default to another color
          if (currentBg !== 0 || (currentInverse && currentFg !== 0)) {
            offset = rectangleCount++ * INDICES_PER_RECTANGLE;
            this._updateRectangle(vertices, offset, currentFg, currentBg, currentStartX, x, y);
          }
          currentStartX = x;
          currentBg = bg;
          currentFg = fg;
          currentInverse = inverse;
        }
      }
      // Finish rectangle if it's still going
      if (currentBg !== 0 || (currentInverse && currentFg !== 0)) {
        offset = rectangleCount++ * INDICES_PER_RECTANGLE;
        this._updateRectangle(vertices, offset, currentFg, currentBg, currentStartX, terminal.cols, y);
      }
    }
    vertices.count = rectangleCount;
  }

  public updateCursor(model: IRenderModel): void {
    const vertices = this._verticesCursor;
    const cursor = model.cursor;
    if (!cursor || cursor.style === 'block') {
      vertices.count = 0;
      return;
    }

    let offset: number;
    let rectangleCount = 0;

    if (cursor.style === 'bar' || cursor.style === 'outline') {
      // Left edge
      offset = rectangleCount++ * INDICES_PER_RECTANGLE;
      this._addRectangleFloat(
        vertices.attributes,
        offset,
        cursor.x * this._dimensions.device.cell.width,
        cursor.y * this._dimensions.device.cell.height,
        cursor.style === 'bar' ? cursor.dpr * cursor.cursorWidth : cursor.dpr,
        this._dimensions.device.cell.height,
        this._cursorFloat
      );
    }
    if (cursor.style === 'underline' || cursor.style === 'outline') {
      // Bottom edge
      offset = rectangleCount++ * INDICES_PER_RECTANGLE;
      this._addRectangleFloat(
        vertices.attributes,
        offset,
        cursor.x * this._dimensions.device.cell.width,
        (cursor.y + 1) * this._dimensions.device.cell.height - cursor.dpr,
        cursor.width * this._dimensions.device.cell.width,
        cursor.dpr,
        this._cursorFloat
      );
    }
    if (cursor.style === 'outline') {
      // Top edge
      offset = rectangleCount++ * INDICES_PER_RECTANGLE;
      this._addRectangleFloat(
        vertices.attributes,
        offset,
        cursor.x * this._dimensions.device.cell.width,
        cursor.y * this._dimensions.device.cell.height,
        cursor.width * this._dimensions.device.cell.width,
        cursor.dpr,
        this._cursorFloat
      );
      // Right edge
      offset = rectangleCount++ * INDICES_PER_RECTANGLE;
      this._addRectangleFloat(
        vertices.attributes,
        offset,
        (cursor.x + cursor.width) * this._dimensions.device.cell.width - cursor.dpr,
        cursor.y * this._dimensions.device.cell.height,
        cursor.dpr,
        this._dimensions.device.cell.height,
        this._cursorFloat
      );
    }

    vertices.count = rectangleCount;
  }

  private _updateRectangle(vertices: Vertices, offset: number, fg: number, bg: number, startX: number, endX: number, y: number): void {
    if (fg & FgFlags.INVERSE) {
      switch (fg & Attributes.CM_MASK) {
        case Attributes.CM_P16:
        case Attributes.CM_P256:
          $rgba = this._themeService.colors.ansi[fg & Attributes.PCOLOR_MASK].rgba;
          break;
        case Attributes.CM_RGB:
          $rgba = (fg & Attributes.RGB_MASK) << 8;
          break;
        case Attributes.CM_DEFAULT:
        default:
          $rgba = this._themeService.colors.foreground.rgba;
      }
    } else {
      switch (bg & Attributes.CM_MASK) {
        case Attributes.CM_P16:
        case Attributes.CM_P256:
          $rgba = this._themeService.colors.ansi[bg & Attributes.PCOLOR_MASK].rgba;
          break;
        case Attributes.CM_RGB:
          $rgba = (bg & Attributes.RGB_MASK) << 8;
          break;
        case Attributes.CM_DEFAULT:
        default:
          $rgba = this._themeService.colors.background.rgba;
      }
    }

    if (vertices.attributes.length < offset + 4) {
      vertices.attributes = expandFloat32Array(vertices.attributes, this._terminal.rows * this._terminal.cols * INDICES_PER_RECTANGLE);
    }
    $x1 = startX * this._dimensions.device.cell.width;
    $y1 = y * this._dimensions.device.cell.height;
    $r = (($rgba >> 24) & 0xFF) / 255;
    $g = (($rgba >> 16) & 0xFF) / 255;
    $b = (($rgba >> 8 ) & 0xFF) / 255;
    $a = 1;

    this._addRectangle(vertices.attributes, offset, $x1, $y1, (endX - startX) * this._dimensions.device.cell.width, this._dimensions.device.cell.height, $r, $g, $b, $a);
  }

  private _addRectangle(array: Float32Array, offset: number, x1: number, y1: number, width: number, height: number, r: number, g: number, b: number, a: number): void {
    array[offset    ] = x1 / this._dimensions.device.canvas.width;
    array[offset + 1] = y1 / this._dimensions.device.canvas.height;
    array[offset + 2] = width / this._dimensions.device.canvas.width;
    array[offset + 3] = height / this._dimensions.device.canvas.height;
    array[offset + 4] = r;
    array[offset + 5] = g;
    array[offset + 6] = b;
    array[offset + 7] = a;
  }

  private _addRectangleFloat(array: Float32Array, offset: number, x1: number, y1: number, width: number, height: number, color: Float32Array): void {
    array[offset    ] = x1 / this._dimensions.device.canvas.width;
    array[offset + 1] = y1 / this._dimensions.device.canvas.height;
    array[offset + 2] = width / this._dimensions.device.canvas.width;
    array[offset + 3] = height / this._dimensions.device.canvas.height;
    array[offset + 4] = color[0];
    array[offset + 5] = color[1];
    array[offset + 6] = color[2];
    array[offset + 7] = color[3];
  }

  private _colorToFloat32Array(color: IColor): Float32Array {
    return new Float32Array([
      ((color.rgba >> 24) & 0xFF) / 255,
      ((color.rgba >> 16) & 0xFF) / 255,
      ((color.rgba >> 8 ) & 0xFF) / 255,
      ((color.rgba      ) & 0xFF) / 255
    ]);
  }
}
