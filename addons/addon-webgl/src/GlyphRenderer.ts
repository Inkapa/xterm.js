/**
 * Copyright (c) 2018 The xterm.js authors. All rights reserved.
 * @license MIT
 */

import { allowRescaling, throwIfFalsy } from 'browser/renderer/shared/RendererUtils';
import { TextureAtlas } from 'browser/renderer/shared/TextureAtlas';
import { IRasterizedGlyph, IRenderDimensions, ITextureAtlas } from 'browser/renderer/shared/Types';
import { NULL_CELL_CODE } from 'common/buffer/Constants';
import { Disposable, toDisposable } from 'common/Lifecycle';
import { Terminal } from '@xterm/xterm';
import { IRenderModel, IWebGL2RenderingContext, IWebGLVertexArrayObject } from './Types';
import { createProgram, GLTexture, PROJECTION_MATRIX } from './WebglUtils';
import type { IOptionsService } from 'common/services/Services';

interface IVertices {
  attributes: Float32Array;
  /**
   * These buffers are the ones used to bind to WebGL, the reason there are
   * multiple is to allow double buffering to work as you cannot modify the
   * buffer while it's being used by the GPU. Having multiple lets us start
   * working on the next frame.
   */
  attributesBuffers: Float32Array[];
  count: number;
}

const enum VertexAttribLocations {
  UNIT_QUAD = 0,
  CELL_POSITION = 1,
  OFFSET = 2,
  SIZE = 3,
  TEXPAGE = 4,
  TEXCOORD = 5,
  TEXSIZE = 6
}

const vertexShaderSource = `#version 300 es
layout (location = ${VertexAttribLocations.UNIT_QUAD}) in vec2 a_unitquad;
layout (location = ${VertexAttribLocations.CELL_POSITION}) in vec2 a_cellpos;
layout (location = ${VertexAttribLocations.OFFSET}) in vec2 a_offset;
layout (location = ${VertexAttribLocations.SIZE}) in vec2 a_size;
layout (location = ${VertexAttribLocations.TEXPAGE}) in float a_texpage;
layout (location = ${VertexAttribLocations.TEXCOORD}) in vec2 a_texcoord;
layout (location = ${VertexAttribLocations.TEXSIZE}) in vec2 a_texsize;

uniform mat4 u_projection;
uniform vec2 u_resolution;
uniform highp int u_vtx;
uniform highp float u_vtime;

out vec2 v_texcoord;
flat out int v_texpage;

// warpVertex distorts the whole frame at the vertex stage
// (window.glyph.vtx). Unlike the per-cell geometry pass these are smooth,
// screen-coherent warps: every quad shares them, so the foreground and the
// background (which runs the identical block) bend together. Applied in
// clip space, which is NDC here because the projection is orthographic.
vec4 warpVertex(vec4 pos) {
  vec2 p = pos.xy;
  if (u_vtx == 1) {
    // barrel: CRT bulge, corners pushed outward
    pos.xy = p * (1.0 + 0.18 * dot(p, p));
  } else if (u_vtx == 2) {
    // shear: the screen skews sideways with height
    pos.x += p.y * 0.25;
  } else if (u_vtx == 3) {
    // quake: the whole image shakes on its own
    pos.xy += vec2(sin(u_vtime * 1.7), cos(u_vtime * 2.3)) * 0.02;
  } else if (u_vtx == 4) {
    // ripple: a travelling horizontal wave, like a flapping flag
    pos.x += sin(p.y * 10.0 + u_vtime * 0.2) * 0.03;
  } else if (u_vtx == 5) {
    // pinch: the centre gets sucked inward
    pos.xy = p * (1.0 - 0.25 * exp(-dot(p, p) * 3.0));
  } else if (u_vtx == 6) {
    // twist: a swirl whose angle grows with radius and drifts in time
    float a = length(p) * 0.8 + u_vtime * 0.01;
    float s = sin(a), c = cos(a);
    pos.xy = mat2(c, -s, s, c) * p;
  } else if (u_vtx == 7) {
    // roll: the picture slides down and snaps back, a vertical hold desync
    pos.y += mod(u_vtime * 0.02, 2.0) - 1.0;
  }
  return pos;
}

void main() {
  vec2 zeroToOne = (a_offset / u_resolution) + a_cellpos + (a_unitquad * a_size);
  gl_Position = warpVertex(u_projection * vec4(zeroToOne, 0.0, 1.0));
  v_texpage = int(a_texpage);
  v_texcoord = a_texcoord + a_unitquad * a_texsize;
}`;

function createFragmentShaderSource(maxFragmentShaderTextureUnits: number): string {
  let textureConditionals = '';
  for (let i = 1; i < maxFragmentShaderTextureUnits; i++) {
    textureConditionals += ` else if (v_texpage == ${i}) { return texture(u_texture[${i}], uv); }`;
  }
  return (`#version 300 es
precision lowp float;

in vec2 v_texcoord;
flat in int v_texpage;

uniform sampler2D u_texture[${maxFragmentShaderTextureUnits}];
uniform highp int u_badgl;
uniform highp float u_time;
uniform highp vec2 u_resolution;

out vec4 outColor;

// Sample the atlas with a constant-index chain: GLSL ES 3.00 requires
// sampler array indices to be constant integral expressions, so the page
// selection cannot be a dynamic index.
vec4 sampleAt(vec2 uv) {
  if (v_texpage == 0) {
    return texture(u_texture[0], uv);
  } ${textureConditionals}
  return vec4(0.0);
}

// 4x4 Bayer ordered-dither threshold matrix, normalised to [0,1). Ordered
// dithering is the single-pass equivalent of error diffusion: each pixel
// picks its output level by comparing against a fixed screen-space
// threshold, so it stays O(1) per fragment with no extra samples. True
// Floyd-Steinberg diffuses each pixel's error to its neighbours, which is
// serial and cannot run in one fragment pass.
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
  outColor = sampleAt(v_texcoord);

  // bad-GL post-processing (window.glyph.shader), animated by u_time.
  if (u_badgl == 1) {
    // static: animated hash noise over the whole frame
    float n = fract(sin(dot(gl_FragCoord.xy + vec2(u_time * 91.7, u_time * 47.3), vec2(12.9898, 78.233))) * 43758.5453);
    outColor.rgb = mix(outColor.rgb, vec3(n), 0.35);
  } else if (u_badgl == 2) {
    // lines: tearing rows, displaced samples
    float h = fract(sin(gl_FragCoord.y * 12.9898 + u_time * 53.0) * 43758.5453);
    if (h > 0.7) {
      vec2 d = vec2((h - 0.85) * 0.2, 0.0);
      outColor = sampleAt(v_texcoord + d);
    }
  } else if (u_badgl == 3) {
    // scan: darken alternating rows
    if (mod(floor(gl_FragCoord.y), 2.0) < 1.0) {
      outColor.rgb *= 0.55;
    }
  } else if (u_badgl == 4) {
    // chan: swap the colour channels
    outColor.rgb = outColor.gbr;
  } else if (u_badgl == 5) {
    // neg: invert
    outColor = vec4(1.0) - outColor;
  } else if (u_badgl == 6) {
    // fade: translucent glyphs
    outColor.a *= 0.55;
  } else if (u_badgl == 7) {
    // flicker: brightness pulses softly
    outColor.rgb *= 0.92 + 0.08 * sin(u_time * 7.0 + gl_FragCoord.x * 0.13);
  } else if (u_badgl == 8) {
    // chroma: channel separation, like an untuned CRT
    vec2 off = vec2(sin(u_time * 5.0) * 0.004, cos(u_time * 3.7) * 0.003);
    outColor.r = sampleAt(v_texcoord - off).r;
    outColor.b = sampleAt(v_texcoord + off).b;
  } else if (u_badgl == 9) {
    // vignette: darken toward the screen edges
    vec2 uv = gl_FragCoord.xy / u_resolution;
    float d = distance(uv, vec2(0.5));
    outColor.rgb *= 1.0 - smoothstep(0.35, 0.78, d) * 0.6;
  } else if (u_badgl == 10) {
    // wave: the sampled glyph ripples through a moving sine warp
    vec2 w = vec2(sin(v_texcoord.y * 60.0 + u_time * 0.2) * 0.006,
                  cos(v_texcoord.x * 60.0 + u_time * 0.17) * 0.006);
    outColor = sampleAt(v_texcoord + w);
  } else if (u_badgl == 11) {
    // mosaic: snap texture coordinates to a coarse grid, chunky pixels
    vec2 grid = vec2(0.012);
    outColor = sampleAt(floor(v_texcoord / grid) * grid);
  } else if (u_badgl == 12) {
    // echo: a ghost of the glyph, offset and added, a smeared trail
    outColor.rgb += sampleAt(v_texcoord - vec2(0.01, 0.0)).rgb * 0.6;
  } else if (u_badgl == 13) {
    // bleed: horizontal RGB smear, the channels run to the right
    outColor.r = sampleAt(v_texcoord - vec2(0.004, 0.0)).r;
    outColor.g = sampleAt(v_texcoord - vec2(0.008, 0.0)).g;
    outColor.b = sampleAt(v_texcoord - vec2(0.012, 0.0)).b;
  } else if (u_badgl == 14) {
    // dither: 1-bit ordered dithering, luminance thresholded against the
    // Bayer matrix so the glyph breaks into a black/white stipple
    float lum = dot(outColor.rgb, vec3(0.299, 0.587, 0.114));
    outColor.rgb = vec3(step(ditherThreshold(gl_FragCoord.xy), lum));
  } else if (u_badgl == 15) {
    // bayer: colour ordered dither, each channel quantised to 4 levels with
    // the threshold nudging the rounding so the banding stipples instead
    float d = ditherThreshold(gl_FragCoord.xy) - 0.5;
    outColor.rgb = clamp(floor(outColor.rgb * 3.0 + 0.5 + d), 0.0, 3.0) / 3.0;
  }
}`);
}

const INDICES_PER_CELL = 11;
const BYTES_PER_CELL = INDICES_PER_CELL * Float32Array.BYTES_PER_ELEMENT;
const CELL_POSITION_INDICES = 2;

// Work variables to avoid garbage collection
let $i = 0;
let $glyph: IRasterizedGlyph | undefined = undefined;
let $leftCellPadding = 0;
let $clippedPixels = 0;

// shaderModeValue maps the page's shader-mode name onto the fragment
// shader's u_badgl uniform.
function shaderModeValue(name: string | undefined): number {
  switch (name) {
    case 'static': return 1;
    case 'lines': return 2;
    case 'scan': return 3;
    case 'chan': return 4;
    case 'neg': return 5;
    case 'fade': return 6;
    case 'flicker': return 7;
    case 'chroma': return 8;
    case 'vignette': return 9;
    case 'wave': return 10;
    case 'mosaic': return 11;
    case 'echo': return 12;
    case 'bleed': return 13;
    case 'dither': return 14;
    case 'bayer': return 15;
    default: return 0;
  }
}

// vtxModeValue maps the page's vertex-warp name onto the u_vtx uniform,
// shared by the glyph and rectangle vertex shaders so the whole frame
// warps coherently.
export function vtxModeValue(name: string | undefined): number {
  switch (name) {
    case 'barrel': return 1;
    case 'shear': return 2;
    case 'quake': return 3;
    case 'ripple': return 4;
    case 'pinch': return 5;
    case 'twist': return 6;
    case 'roll': return 7;
    default: return 0;
  }
}

// The glyph control surface: window.glyph, created by the page. Absent in
// tests and non-browser contexts, where the modes stay off.
function glyphCfg(): any {
  return (typeof globalThis !== 'undefined' && (globalThis as any).glyph) || {};
}

export class GlyphRenderer extends Disposable {
  private readonly _program: WebGLProgram;
  private readonly _vertexArrayObject: IWebGLVertexArrayObject;
  private readonly _projectionLocation: WebGLUniformLocation;
  private readonly _resolutionLocation: WebGLUniformLocation;
  private readonly _textureLocation: WebGLUniformLocation;
  private readonly _badglLocation: WebGLUniformLocation;
  private readonly _timeLocation: WebGLUniformLocation;
  private readonly _fragResolutionLocation: WebGLUniformLocation;
  private readonly _vtxLocation: WebGLUniformLocation;
  private readonly _vtimeLocation: WebGLUniformLocation;
  private readonly _atlasTextures: GLTexture[];
  private readonly _attributesBuffer: WebGLBuffer;

  private _atlas: ITextureAtlas | undefined;
  private _activeBuffer: number = 0;
  private _frameCount: number = 0;
  private readonly _vertices: IVertices = {
    count: 0,
    attributes: new Float32Array(0),
    attributesBuffers: [
      new Float32Array(0),
      new Float32Array(0)
    ]
  };

  constructor(
    private readonly _terminal: Terminal,
    private readonly _gl: IWebGL2RenderingContext,
    private _dimensions: IRenderDimensions,
    private readonly _optionsService: IOptionsService
  ) {
    super();

    const gl = this._gl;

    if (TextureAtlas.maxAtlasPages === undefined) {
      // Typically 8 or 16
      TextureAtlas.maxAtlasPages = Math.min(32, throwIfFalsy(gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS) as number | null));
      // Almost all clients will support >= 4096
      TextureAtlas.maxTextureSize = throwIfFalsy(gl.getParameter(gl.MAX_TEXTURE_SIZE) as number | null);
    }

    this._program = throwIfFalsy(createProgram(gl, vertexShaderSource, createFragmentShaderSource(TextureAtlas.maxAtlasPages)));
    this.register(toDisposable(() => gl.deleteProgram(this._program)));

    // Uniform locations
    this._projectionLocation = throwIfFalsy(gl.getUniformLocation(this._program, 'u_projection'));
    this._resolutionLocation = throwIfFalsy(gl.getUniformLocation(this._program, 'u_resolution'));
    this._textureLocation = throwIfFalsy(gl.getUniformLocation(this._program, 'u_texture'));
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
    gl.enableVertexAttribArray(VertexAttribLocations.OFFSET);
    gl.vertexAttribPointer(VertexAttribLocations.OFFSET, 2, gl.FLOAT, false, BYTES_PER_CELL, 0);
    gl.vertexAttribDivisor(VertexAttribLocations.OFFSET, 1);
    gl.enableVertexAttribArray(VertexAttribLocations.SIZE);
    gl.vertexAttribPointer(VertexAttribLocations.SIZE, 2, gl.FLOAT, false, BYTES_PER_CELL, 2 * Float32Array.BYTES_PER_ELEMENT);
    gl.vertexAttribDivisor(VertexAttribLocations.SIZE, 1);
    gl.enableVertexAttribArray(VertexAttribLocations.TEXPAGE);
    gl.vertexAttribPointer(VertexAttribLocations.TEXPAGE, 1, gl.FLOAT, false, BYTES_PER_CELL, 4 * Float32Array.BYTES_PER_ELEMENT);
    gl.vertexAttribDivisor(VertexAttribLocations.TEXPAGE, 1);
    gl.enableVertexAttribArray(VertexAttribLocations.TEXCOORD);
    gl.vertexAttribPointer(VertexAttribLocations.TEXCOORD, 2, gl.FLOAT, false, BYTES_PER_CELL, 5 * Float32Array.BYTES_PER_ELEMENT);
    gl.vertexAttribDivisor(VertexAttribLocations.TEXCOORD, 1);
    gl.enableVertexAttribArray(VertexAttribLocations.TEXSIZE);
    gl.vertexAttribPointer(VertexAttribLocations.TEXSIZE, 2, gl.FLOAT, false, BYTES_PER_CELL, 7 * Float32Array.BYTES_PER_ELEMENT);
    gl.vertexAttribDivisor(VertexAttribLocations.TEXSIZE, 1);
    gl.enableVertexAttribArray(VertexAttribLocations.CELL_POSITION);
    gl.vertexAttribPointer(VertexAttribLocations.CELL_POSITION, 2, gl.FLOAT, false, BYTES_PER_CELL, 9 * Float32Array.BYTES_PER_ELEMENT);
    gl.vertexAttribDivisor(VertexAttribLocations.CELL_POSITION, 1);

    // Setup static uniforms
    gl.useProgram(this._program);
    const textureUnits = new Int32Array(TextureAtlas.maxAtlasPages);
    for (let i = 0; i < TextureAtlas.maxAtlasPages; i++) {
      textureUnits[i] = i;
    }
    gl.uniform1iv(this._textureLocation, textureUnits);
    gl.uniformMatrix4fv(this._projectionLocation, false, PROJECTION_MATRIX);

    // Setup 1x1 red pixel textures for all potential atlas pages, if one of these invalid textures
    // is ever drawn it will show characters as red rectangles.
    this._atlasTextures = [];
    for (let i = 0; i < TextureAtlas.maxAtlasPages; i++) {
      const glTexture = new GLTexture(throwIfFalsy(gl.createTexture()));
      this.register(toDisposable(() => gl.deleteTexture(glTexture.texture)));
      gl.activeTexture(gl.TEXTURE0 + i);
      gl.bindTexture(gl.TEXTURE_2D, glTexture.texture);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([255, 0, 0, 255]));
      this._atlasTextures[i] = glTexture;
    }

    // Allow drawing of transparent texture
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    // Set viewport
    this.handleResize();
  }

  public beginFrame(): boolean {
    return this._atlas ? this._atlas.beginFrame() : true;
  }

  public updateCell(x: number, y: number, code: number, bg: number, fg: number, ext: number, chars: string, width: number, lastBg: number): void {
    // Since this function is called for every cell (`rows*cols`), it must be very optimized. It
    // should not instantiate any variables unless a new glyph is drawn to the cache where the
    // slight slowdown is acceptable for the developer ergonomics provided as it's a once of for
    // each glyph.
    this._updateCell(this._vertices.attributes, x, y, code, bg, fg, ext, chars, width, lastBg);
  }

  private _updateCell(array: Float32Array, x: number, y: number, code: number | undefined, bg: number, fg: number, ext: number, chars: string, width: number, lastBg: number): void {
    $i = (y * this._terminal.cols + x) * INDICES_PER_CELL;

    // Exit early if this is a null character, allow space character to continue as it may have
    // underline/strikethrough styles
    if (code === NULL_CELL_CODE || code === undefined/* This is used for the right side of wide chars */) {
      array.fill(0, $i, $i + INDICES_PER_CELL - 1 - CELL_POSITION_INDICES);
      return;
    }

    if (!this._atlas) {
      return;
    }

    // Get the glyph
    if (chars && chars.length > 1) {
      $glyph = this._atlas.getRasterizedGlyphCombinedChar(chars, bg, fg, ext, false);
    } else {
      $glyph = this._atlas.getRasterizedGlyph(code, bg, fg, ext, false);
    }

    $leftCellPadding = Math.floor((this._dimensions.device.cell.width - this._dimensions.device.char.width) / 2);
    if (bg !== lastBg && $glyph.offset.x > $leftCellPadding) {
      // The glyph may carry a stale page index from a mid-frame merge;
      // the next frame reindexes, so drop the cell instead of throwing.
      const texPage = this._atlas.pages[$glyph.texturePage];
      if (!texPage) {
        array.fill(0, $i, $i + INDICES_PER_CELL - 1 - CELL_POSITION_INDICES);
        return;
      }
      $clippedPixels = $glyph.offset.x - $leftCellPadding;
      // a_origin
      array[$i    ] = -($glyph.offset.x - $clippedPixels) + this._dimensions.device.char.left;
      array[$i + 1] = -$glyph.offset.y + this._dimensions.device.char.top;
      // a_size
      array[$i + 2] = ($glyph.size.x - $clippedPixels) / this._dimensions.device.canvas.width;
      array[$i + 3] = $glyph.size.y / this._dimensions.device.canvas.height;
      // a_texpage
      array[$i + 4] = $glyph.texturePage;
      // a_texcoord
      array[$i + 5] = $glyph.texturePositionClipSpace.x + $clippedPixels / texPage.canvas.width;
      array[$i + 6] = $glyph.texturePositionClipSpace.y;
      // a_texsize
      array[$i + 7] = $glyph.sizeClipSpace.x - $clippedPixels / texPage.canvas.width;
      array[$i + 8] = $glyph.sizeClipSpace.y;
    } else {
      // a_origin
      array[$i    ] = -$glyph.offset.x + this._dimensions.device.char.left;
      array[$i + 1] = -$glyph.offset.y + this._dimensions.device.char.top;
      // a_size
      array[$i + 2] = $glyph.size.x / this._dimensions.device.canvas.width;
      array[$i + 3] = $glyph.size.y / this._dimensions.device.canvas.height;
      // a_texpage
      array[$i + 4] = $glyph.texturePage;
      // a_texcoord
      array[$i + 5] = $glyph.texturePositionClipSpace.x;
      array[$i + 6] = $glyph.texturePositionClipSpace.y;
      // a_texsize
      array[$i + 7] = $glyph.sizeClipSpace.x;
      array[$i + 8] = $glyph.sizeClipSpace.y;
    }
    // a_cellpos only changes on resize

    // Reduce scale horizontally for wide glyphs printed in cells that would overlap with the
    // following cell (ie. the width is not 2).
    if (this._optionsService.rawOptions.rescaleOverlappingGlyphs) {
      if (allowRescaling(code, width, $glyph.size.x, this._dimensions.device.cell.width)) {
        array[$i + 2] = (this._dimensions.device.cell.width - 1) / this._dimensions.device.canvas.width; // - 1 to improve readability
      }
    }

    // glyph bad-GL modes (window.glyph.badgl): deliberate renderer
    // corruption, read live per cell. No atlas or shader state is
    // touched, so the modes cannot break the page arrays.
    const g = glyphCfg();
    const hm = (x * 73856093) ^ (y * 19349663) ^ (bg >>> 13) ^ (fg >>> 7);
    const hA = (hm & 0xFFFF) / 0xFFFF;
    const hB = ((hm >>> 16) & 0xFFFF) / 0xFFFF;
    const cw = this._dimensions.device.char.width;
    const ch = this._dimensions.device.char.height;
    let mode = g.badgl;
    if (mode === 'mix') {
      mode = ['jitter', 'page', 'tex', 'cut', 'stretch', 'shift', 'flip', 'skip', 'zebra', 'block', 'band', 'drift', 'wobble', 'squint', 'snow', 'melt', 'tear', 'throb', 'explode', 'spike', 'crush'][Math.floor(hA * 21)];
    }
    switch (mode) {
      case 'jitter':
        array[$i] += (hA - 0.5) * 2 * cw * 2;
        array[$i + 1] += (hB - 0.5) * 2 * ch * 2;
        break;
      case 'page':
        // Draw each cell from the next atlas page, clamped to the bound
        // sampler array: an index past it samples an unbound texture and
        // renders black.
        array[$i + 4] = (array[$i + 4] + 1) % Math.max(1, Math.min(this._atlas.pages.length, this._atlasTextures.length));
        break;
      case 'tex':
        array[$i + 5] += (hA - 0.5) * 0.5;
        array[$i + 6] += (hB - 0.5) * 0.5;
        break;
      case 'cut':
        // half-cut characters: the quad is clipped top or bottom
        if (hA < 0.5) {
          array[$i + 3] *= 0.5;
        } else {
          array[$i + 1] += ch / 2;
          array[$i + 3] *= 0.5;
        }
        break;
      case 'stretch':
        array[$i + 2] *= 0.6 + 1.4 * hA;
        array[$i + 3] *= 0.6 + 0.8 * hB;
        break;
      case 'shift':
        // glyphs sit between cells and get cut at the cell edges
        array[$i] += cw * (hA < 0.5 ? -0.5 : 0.5);
        array[$i + 1] += ch * (hB < 0.5 ? -0.5 : 0.5);
        break;
      case 'flip':
        // mirrored glyphs: negative texture span with an offset base
        if (hA < 0.5) {
          array[$i + 5] += array[$i + 7];
          array[$i + 7] = -array[$i + 7];
        } else {
          array[$i + 6] += array[$i + 8];
          array[$i + 8] = -array[$i + 8];
        }
        break;
      case 'skip':
        // cells vanish: zero the vertex, the cell draws nothing
        if (hA < 0.25) {
          array.fill(0, $i, $i + INDICES_PER_CELL - 1 - CELL_POSITION_INDICES);
          return;
        }
        break;
      case 'zebra':
        // checkerboard of wrong texture regions
        if (((x + y) & 1) === 0) {
          array[$i + 5] += 0.25 * (hA - 0.5);
          array[$i + 6] += 0.25 * (hB - 0.5);
        }
        break;
      case 'block':
        // whole 2x2 blocks corrupt together
        if ((((x >> 1) * 73856093) ^ ((y >> 1) * 19349663) ^ (bg >>> 13) ^ (fg >>> 7) & 0xFFFF) / 0xFFFF < 0.4) {
          array[$i + 5] += 0.4 * (hA - 0.5);
          array[$i + 6] += 0.4 * (hB - 0.5);
        }
        break;
      case 'band':
        // wobbling stripes of garbage, the band position per column
        // shifts with the cell colours
        if (((y + Math.floor(hB * 8)) % 8) < 3) {
          array[$i + 5] += 0.6 * (hA - 0.5);
        }
        break;      case 'drift':
        // sub-pixel offset drift: glyphs sit slightly off, fuzzy edges
        array[$i] += (hA - 0.5) * cw * 0.25;
        array[$i + 1] += (hB - 0.5) * ch * 0.25;
        break;
      case 'wobble':
        // gentle per-column wave
        array[$i] += Math.sin(y * 0.35 + hB * 6.28) * cw * 0.3;
        break;
      case 'squint':
        // glyphs slightly compressed
        array[$i + 3] *= 0.85;
        break;
      case 'snow':
        // rare cells nudge slightly off their texture region
        if (hA < 0.05) {
          array[$i + 5] += (hB - 0.5) * 0.2;
        }
        break;
      case 'melt':
        // time-driven drip: each cell slides downward at a speed set by
        // its hash and wraps after a few rows, so the screen runs like
        // wet ink and the layout never settles.
        array[$i + 1] += (this._frameCount * (0.4 + hA) * 2 + hB * ch * 8) % (ch * 8);
        break;
      case 'tear':
        // horizontal tracking tear: whole rows jump sideways together,
        // the shift per row wandering with time like a mistuned VHS head.
        {
          const t = Math.sin(y * 0.7 + this._frameCount * 0.08);
          if (t > 0.6) {
            array[$i] += (t - 0.6) * cw * 20;
          }
        }
        break;
      case 'throb':
        // time-driven breathing: every cell pulses in size, overshooting
        // 1 so glyphs swell over their neighbours and contract to nothing.
        {
          const p = 0.5 + 1.1 * (0.5 + 0.5 * Math.sin(this._frameCount * 0.15 + hA * 6.28));
          array[$i + 2] *= p;
          array[$i + 3] *= p;
        }
        break;
      case 'explode':
        // radial blast from the screen centre: glyphs fly outward, the
        // far ones leaving the viewport entirely.
        {
          const dx = (x / this._terminal.cols) - 0.5;
          const dy = (y / this._terminal.rows) - 0.5;
          array[$i]     += dx * cw * (4 + hA * 12);
          array[$i + 1] += dy * ch * (4 + hB * 12);
        }
        break;
      case 'spike':
        // rare cells detonate to many times their size: one stamp smeared
        // across a whole region, overflowing its neighbours.
        if (hA < 0.03) {
          array[$i + 2] *= 6 + hB * 10;
          array[$i + 3] *= 6 + hA * 10;
        }
        break;
      case 'crush':
        // random rows of cells collapse to a single scanline, a
        // dying-CRT horizontal streak.
        if (hB < 0.3) {
          array[$i + 3] *= 0.06;
          array[$i + 1] += ch * 0.5;
        }
        break;
    }
  }

  public clear(): void {
    const terminal = this._terminal;
    const newCount = terminal.cols * terminal.rows * INDICES_PER_CELL;

    // Clear vertices
    if (this._vertices.count !== newCount) {
      this._vertices.attributes = new Float32Array(newCount);
    } else {
      this._vertices.attributes.fill(0);
    }
    let i = 0;
    for (; i < this._vertices.attributesBuffers.length; i++) {
      if (this._vertices.count !== newCount) {
        this._vertices.attributesBuffers[i] = new Float32Array(newCount);
      } else {
        this._vertices.attributesBuffers[i].fill(0);
      }
    }
    this._vertices.count = newCount;
    i = 0;
    for (let y = 0; y < terminal.rows; y++) {
      for (let x = 0; x < terminal.cols; x++) {
        this._vertices.attributes[i + 9] = x / terminal.cols;
        this._vertices.attributes[i + 10] = y / terminal.rows;
        i += INDICES_PER_CELL;
      }
    }
  }

  public handleResize(): void {
    const gl = this._gl;
    gl.useProgram(this._program);
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
    gl.uniform2f(this._resolutionLocation, gl.canvas.width, gl.canvas.height);
    gl.uniform2f(this._fragResolutionLocation, gl.canvas.width, gl.canvas.height);
    this.clear();
  }

  public render(renderModel: IRenderModel): void {
    if (!this._atlas) {
      return;
    }

    const gl = this._gl;

    gl.useProgram(this._program);
    gl.bindVertexArray(this._vertexArrayObject);

    // bad-GL shader modes, read live per frame: the mode uniform plus an
    // animated time value for the post-processing block.
    gl.uniform1i(this._badglLocation, shaderModeValue(glyphCfg().shader));
    gl.uniform1f(this._timeLocation, this._frameCount);
    gl.uniform1i(this._vtxLocation, vtxModeValue(glyphCfg().vtx));
    gl.uniform1f(this._vtimeLocation, this._frameCount);
    this._frameCount++;

    // Alternate buffers each frame as the active buffer gets locked while it's in use by the GPU
    this._activeBuffer = (this._activeBuffer + 1) % 2;
    const activeBuffer = this._vertices.attributesBuffers[this._activeBuffer];

    // Copy data for each cell of each line up to its line length (the last non-whitespace cell)
    // from the attributes buffer into activeBuffer, which is the one that gets bound to the GPU.
    // The reasons for this are as follows:
    // - So the active buffer can be alternated so we don't get blocked on rendering finishing
    // - To copy either the normal attributes buffer or the selection attributes buffer when there
    //   is a selection
    // - So we don't send vertices for all the line-ending whitespace to the GPU
    let bufferLength = 0;
    for (let y = 0; y < renderModel.lineLengths.length; y++) {
      const si = y * this._terminal.cols * INDICES_PER_CELL;
      const sub = this._vertices.attributes.subarray(si, si + renderModel.lineLengths[y] * INDICES_PER_CELL);
      activeBuffer.set(sub, bufferLength);
      bufferLength += sub.length;
    }

    // Bind the attributes buffer
    gl.bindBuffer(gl.ARRAY_BUFFER, this._attributesBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, activeBuffer.subarray(0, bufferLength), gl.STREAM_DRAW);

    // Bind the atlas page texture if they have changed. The texture array
    // is sized at construction (one entry per shader sampler), while the
    // atlas pages can grow past it under a colour flood; cells on pages
    // beyond the array render as unbound-sampler garbage, which is the
    // corruption, but the loop must not dereference past the array.
    for (let i = 0; i < this._atlas.pages.length && i < this._atlasTextures.length; i++) {
      if (this._atlas.pages[i].version !== this._atlasTextures[i].version) {
        this._bindAtlasPageTexture(gl, this._atlas, i);
      }
    }

    // Draw the viewport
    gl.drawElementsInstanced(gl.TRIANGLE_STRIP, 4, gl.UNSIGNED_BYTE, 0, bufferLength / INDICES_PER_CELL);
  }

  public setAtlas(atlas: ITextureAtlas): void {
    this._atlas = atlas;
    for (const glTexture of this._atlasTextures) {
      glTexture.version = -1;
    }
  }

  private _bindAtlasPageTexture(gl: IWebGL2RenderingContext, atlas: ITextureAtlas, i: number): void {
    gl.activeTexture(gl.TEXTURE0 + i);
    gl.bindTexture(gl.TEXTURE_2D, this._atlasTextures[i].texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, atlas.pages[i].canvas);
    gl.generateMipmap(gl.TEXTURE_2D);
    this._atlasTextures[i].version = atlas.pages[i].version;
  }

  public setDimensions(dimensions: IRenderDimensions): void {
    this._dimensions = dimensions;
  }
}
