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
import { backgroundRgba } from './CellBackground';
import { createProgram, GLTexture, PROJECTION_MATRIX } from './WebglUtils';
import type { IThemeService } from 'browser/services/Services';
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
  TEXSIZE = 6,
  COLOR = 7,
  BGCOLOR = 8
}

const vertexShaderSource = `#version 300 es
layout (location = ${VertexAttribLocations.UNIT_QUAD}) in vec2 a_unitquad;
layout (location = ${VertexAttribLocations.CELL_POSITION}) in vec2 a_cellpos;
layout (location = ${VertexAttribLocations.OFFSET}) in vec2 a_offset;
layout (location = ${VertexAttribLocations.SIZE}) in vec2 a_size;
layout (location = ${VertexAttribLocations.TEXPAGE}) in float a_texpage;
layout (location = ${VertexAttribLocations.TEXCOORD}) in vec2 a_texcoord;
layout (location = ${VertexAttribLocations.TEXSIZE}) in vec2 a_texsize;
layout (location = ${VertexAttribLocations.COLOR}) in vec4 a_color;
layout (location = ${VertexAttribLocations.BGCOLOR}) in vec4 a_bgcolor;

uniform mat4 u_projection;
uniform vec2 u_resolution;
uniform highp int u_vtx;
uniform highp float u_vtime;
uniform highp float u_intensity;
// Per-effect character knobs, each a multiplier defaulting to 1.0 (the
// built-in look). Slot 0 (warp) scales the vertex distortion here; the
// fragment stage reads the rest. See paramValues for the slot mapping.
uniform highp float u_param[4];

out vec2 v_texcoord;
flat out int v_texpage;
out vec4 v_color;
out vec4 v_bgcolor;

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
  vec4 base = u_projection * vec4(zeroToOne, 0.0, 1.0);
  // u_intensity (0..1) scales the warp: 0 leaves the frame undistorted, 1
  // is the full warp, so a tween fades the distortion in and out. u_param[0]
  // (warp) scales the displacement itself, so a driver can push the amount
  // past the built-in look or flatten one warp without touching the master.
  vec4 warped = warpVertex(base);
  gl_Position = mix(base, base + (warped - base) * u_param[0], u_intensity);
  v_texpage = int(a_texpage);
  v_texcoord = a_texcoord + a_unitquad * a_texsize;
  v_color = a_color;
  v_bgcolor = a_bgcolor;
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
in vec4 v_color;
in vec4 v_bgcolor;

uniform sampler2D u_texture[${maxFragmentShaderTextureUnits}];
uniform highp int u_badgl;
uniform highp float u_time;
uniform highp vec2 u_resolution;
uniform highp float u_intensity;
// Per-effect character knobs (see paramValues): [1] chroma spread, [2] echo
// gain, [3] dither level count. Each multiplies one effect's built-in
// constant and defaults to 1.0, so a driver shapes a single filter.
uniform highp float u_param[4];

out vec4 outColor;

// Sample the atlas with a constant-index chain: GLSL ES 3.00 requires
// sampler array indices to be constant integral expressions, so the page
// selection cannot be a dynamic index.
vec4 sampleAtlas(vec2 uv) {
  if (v_texpage == 0) {
    return texture(u_texture[0], uv);
  } ${textureConditionals}
  return vec4(0.0);
}

// Every read of the glyph goes through the cell's colour. A tinted atlas
// holds white coverage masks and v_color is the cell's foreground; a
// colour-baked atlas already holds the colour and v_color is white. Either
// way the filters below resample correctly coloured glyphs.
//
// A pixel-plane cell carries its background too and covers the whole cell:
// the upper half block's mask is 1 across the top and 0 across the bottom,
// so mixing the two colours across it reproduces what the scene painted.
// Those cells draw opaque and get no rectangle behind them, which is what
// lets a per-cell mode move one without leaving a copy of it behind. Every
// filter below is written in terms of this function, the ones that
// resample at an offset included, so none of them change.
vec4 sampleAt(vec2 uv) {
  vec4 texel = sampleAtlas(uv);
  if (v_bgcolor.a > 0.0) {
    return vec4(mix(v_bgcolor.rgb, v_color.rgb, texel.a), 1.0);
  }
  return v_color * texel;
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
  // The unfiltered glyph, kept so u_intensity can crossfade the whole
  // filter stack back toward it (0 = clean text, 1 = full stack).
  vec4 glyphClean = outColor;

  // bad-GL post-processing (window.glyph.shader), animated by u_time.
  // u_badgl is a bitmask, one bit per filter, so several stack in a single
  // pass and compose in ascending bit order. shaderModeValue in the page
  // ORs the requested modes together; a single mode is just one bit. The
  // baseline cost is a handful of bitwise tests; only stacking the
  // atlas-resampling filters (chroma, wave, mosaic, echo, bleed, dither)
  // adds real texture work, so that is where a heavy stack meets its limit.
  if ((u_badgl & 1) != 0) {
    // static: animated hash noise over the whole frame
    float n = fract(sin(dot(gl_FragCoord.xy + vec2(u_time * 91.7, u_time * 47.3), vec2(12.9898, 78.233))) * 43758.5453);
    outColor.rgb = mix(outColor.rgb, vec3(n), 0.35);
  }
  if ((u_badgl & 2) != 0) {
    // lines: tearing rows, displaced samples
    float h = fract(sin(gl_FragCoord.y * 12.9898 + u_time * 53.0) * 43758.5453);
    if (h > 0.7) {
      vec2 d = vec2((h - 0.85) * 0.2, 0.0);
      outColor = sampleAt(v_texcoord + d);
    }
  }
  if ((u_badgl & 4) != 0) {
    // scan: darken alternating rows
    if (mod(floor(gl_FragCoord.y), 2.0) < 1.0) {
      outColor.rgb *= 0.55;
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
    // fade: translucent glyphs
    outColor.a *= 0.55;
  }
  if ((u_badgl & 64) != 0) {
    // flicker: brightness pulses softly
    outColor.rgb *= 0.92 + 0.08 * sin(u_time * 7.0 + gl_FragCoord.x * 0.13);
  }
  if ((u_badgl & 128) != 0) {
    // chroma: channel separation, like an untuned CRT
    vec2 off = vec2(sin(u_time * 5.0) * 0.004, cos(u_time * 3.7) * 0.003) * u_param[1];
    outColor.r = sampleAt(v_texcoord - off).r;
    outColor.b = sampleAt(v_texcoord + off).b;
  }
  if ((u_badgl & 256) != 0) {
    // vignette: darken toward the screen edges
    vec2 uv = gl_FragCoord.xy / u_resolution;
    float d = distance(uv, vec2(0.5));
    outColor.rgb *= 1.0 - smoothstep(0.35, 0.78, d) * 0.6;
  }
  if ((u_badgl & 512) != 0) {
    // wave: the sampled glyph ripples through a moving sine warp
    vec2 w = vec2(sin(v_texcoord.y * 60.0 + u_time * 0.2) * 0.006,
                  cos(v_texcoord.x * 60.0 + u_time * 0.17) * 0.006);
    outColor = sampleAt(v_texcoord + w);
  }
  if ((u_badgl & 1024) != 0) {
    // mosaic: snap texture coordinates to a coarse grid, chunky pixels
    vec2 grid = vec2(0.012);
    outColor = sampleAt(floor(v_texcoord / grid) * grid);
  }
  if ((u_badgl & 2048) != 0) {
    // echo: a ghost of the glyph, offset and added, a smeared trail
    outColor.rgb += sampleAt(v_texcoord - vec2(0.01, 0.0)).rgb * 0.6 * u_param[2];
  }
  if ((u_badgl & 4096) != 0) {
    // bleed: horizontal RGB smear, the channels run to the right. Shares the
    // chroma knob (u_param[1]) since both spread the colour channels.
    outColor.r = sampleAt(v_texcoord - vec2(0.004, 0.0) * u_param[1]).r;
    outColor.g = sampleAt(v_texcoord - vec2(0.008, 0.0) * u_param[1]).g;
    outColor.b = sampleAt(v_texcoord - vec2(0.012, 0.0) * u_param[1]).b;
  }
  if ((u_badgl & 8192) != 0) {
    // dither: 1-bit ordered dithering, luminance thresholded against the
    // Bayer matrix so the glyph breaks into a black/white stipple
    float lum = dot(outColor.rgb, vec3(0.299, 0.587, 0.114));
    outColor.rgb = vec3(step(ditherThreshold(gl_FragCoord.xy), lum));
  }
  if ((u_badgl & 16384) != 0) {
    // bayer: colour ordered dither, each channel quantised with the threshold
    // nudging the rounding so the banding stipples instead. u_param[3] scales
    // the step count (default 3 -> 4 levels): higher is finer, lower coarser.
    float steps = max(1.0, 3.0 * u_param[3]);
    float d = ditherThreshold(gl_FragCoord.xy) - 0.5;
    outColor.rgb = clamp(floor(outColor.rgb * steps + 0.5 + d), 0.0, steps) / steps;
  }

  // Master crossfade: scale the whole filtered result back toward the
  // untouched glyph by u_intensity.
  outColor = mix(glyphClean, outColor, u_intensity);
}`);
}

// Per-cell instanced attributes, in float order:
//   [0,1] offset  [2,3] size  [4] texpage  [5,6] texcoord  [7,8] texsize
//   [9..12] colour (rgba)  [13,14] cellpos
// cellpos is last so a cell that draws nothing can zero everything before it;
// it is written once per resize in clear().
const INDICES_PER_CELL = 19;
const BYTES_PER_CELL = INDICES_PER_CELL * Float32Array.BYTES_PER_ELEMENT;
const CELL_POSITION_INDICES = 2;

// Work variables to avoid garbage collection
let $i = 0;
let $glyph: IRasterizedGlyph | undefined = undefined;
let $leftCellPadding = 0;
let $clippedPixels = 0;
let $fgRgba = 0;
let $bgRgba = 0;

// shaderBit maps one foreground filter name onto its bit in the u_badgl
// mask. The bit order is the order the filters compose in the shader.
function shaderBit(name: string): number {
  switch (name.trim()) {
    case 'static': return 1;
    case 'lines': return 2;
    case 'scan': return 4;
    case 'chan': return 8;
    case 'neg': return 16;
    case 'fade': return 32;
    case 'flicker': return 64;
    case 'chroma': return 128;
    case 'vignette': return 256;
    case 'wave': return 512;
    case 'mosaic': return 1024;
    case 'echo': return 2048;
    case 'bleed': return 4096;
    case 'dither': return 8192;
    case 'bayer': return 16384;
    default: return 0;
  }
}

// shaderModeValue turns the page's shader field into the u_badgl bitmask.
// It accepts a comma-separated list ('chroma,scan,dither'), so several
// filters stack in one pass; a single name is just one bit.
function shaderModeValue(name: string | undefined): number {
  if (!name) {
    return 0;
  }
  let mask = 0;
  for (const part of name.split(',')) {
    mask |= shaderBit(part);
  }
  return mask;
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

// The page's shared per-cell module (glyph's renderers/percell, built to
// WebAssembly), present once the page has loaded it.
export interface IPercellModule {
  active: boolean;
  geometry(cols: number, rows: number, frame: number): Float32Array;
  substitute(code: number, dc: number, dr: number): number;
  /** One code point per cell, row-major, written before geometry is asked for. */
  glyphs(length: number): Uint32Array;
  /** The amounts for one layer: 0 content, 1 pixel. */
  stateFor(layer: number): Float32Array;
  /** The code points that mark the pixel plane, which the shared crate owns. */
  pixelGlyphCount(): number;
  pixelGlyph(i: number): number;
}

// percellModule returns the module while a per-cell mode is on, and
// undefined otherwise: before the page loads it, in tests, and whenever
// every per-cell amount is zero.
export function percellModule(): IPercellModule | undefined {
  const module = glyphCfg().percell as IPercellModule | undefined;
  return module && module.active ? module : undefined;
}

// Channel layout of IPercellModule.geometry, channel-major, one run of
// cols*rows floats per channel. Matches glyph's cell::CellGeometry::channels.
const enum PercellChannel { DX, DY, W_SCALE, H_SCALE, DC, DR, MIRROR_X, MIRROR_Y, ACTIVE, COUNT }

// New glyphs the index modes may rasterize into the atlas per 60 Hz frame's
// worth of time. Unbudgeted substitution at a 211x49 grid measured over 80%
// of the main thread; this bounds it, and the atlas fills up over a few
// frames on steady colours. It refills by elapsed time rather than per
// frame, so a 144 Hz display rasterizes no more per second than a 60 Hz one.
const SUBSTITUTION_BUDGET = 64;
const SUBSTITUTION_REFILL_MS = 1000 / 60;

// intensityValue reads the master glitch intensity (0..1) from the page's
// params bag, clamped. It scales the continuous stages (the vertex warp and
// the fragment filters) so a tween or a reactive driver can fade the whole
// glitch in and out. Defaults to 1 (full) when unset. Exported so the
// rectangle renderer reads the same value and the background matches.
export function intensityValue(): number {
  const p = glyphCfg().params;
  const v = p && typeof p.intensity === 'number' ? p.intensity : 1;
  return v < 0 ? 0 : (v > 1 ? 1 : v);
}

// paramValues fills the shared u_param[] array from the page's params bag.
// Each slot is a per-effect multiplier on that effect's built-in constant and
// defaults to 1.0 (the stock look) when unset, so a tween or a reactive driver
// can shape one effect's character rather than only the master fade. The slots
// are fixed and mirror the shader: 0 warp (vertex distortion amount), 1 chroma
// (RGB-split spread, shared by chroma and bleed), 2 echo (ghost gain), 3
// dither (bayer level count). Exported so both renderers set the same values
// and the foreground and background stay in step. Reuses one array to avoid a
// per-frame allocation, matching the work-variable idiom above.
const PARAM_KEYS = ['warp', 'chroma', 'echo', 'dither'];
const $paramArray = new Float32Array(PARAM_KEYS.length);
export function paramValues(): Float32Array {
  const p = glyphCfg().params || {};
  for (let i = 0; i < PARAM_KEYS.length; i++) {
    const v = p[PARAM_KEYS[i]];
    $paramArray[i] = typeof v === 'number' ? v : 1.0;
  }
  return $paramArray;
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
  private readonly _intensityLocation: WebGLUniformLocation;
  private readonly _paramLocation: WebGLUniformLocation;
  private readonly _atlasTextures: GLTexture[];
  private readonly _attributesBuffer: WebGLBuffer;

  private _atlas: ITextureAtlas | undefined;
  private _activeBuffer: number = 0;
  private _frameCount: number = 0;
  private _percell: IPercellModule | undefined;
  private _pixelGlyphs: Set<number> | undefined;
  private _percellGeometry: Float32Array | undefined;
  private _substitutionBudget: number = 0;
  private _substitutionRefilledAt: number = 0;
  // Cells whose substitute was not in the atlas, counted this frame, and
  // the admission stride worked out from last frame's count.
  private _substitutionMisses: number = 0;
  private _substitutionStride: number = 1;
  private _percellFrame: number = 0;
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
    private readonly _optionsService: IOptionsService,
    private readonly _themeService: IThemeService
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
    // u_intensity is declared in both stages of this program, so it links to
    // one shared location that the vertex and fragment shaders both read.
    this._intensityLocation = throwIfFalsy(gl.getUniformLocation(this._program, 'u_intensity'));
    // u_param[] is declared in both stages and links to one shared location;
    // query the first element for portability across drivers.
    this._paramLocation = throwIfFalsy(gl.getUniformLocation(this._program, 'u_param[0]'));

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
    gl.enableVertexAttribArray(VertexAttribLocations.COLOR);
    gl.vertexAttribPointer(VertexAttribLocations.COLOR, 4, gl.FLOAT, false, BYTES_PER_CELL, 9 * Float32Array.BYTES_PER_ELEMENT);
    gl.vertexAttribDivisor(VertexAttribLocations.COLOR, 1);
    gl.enableVertexAttribArray(VertexAttribLocations.BGCOLOR);
    gl.vertexAttribPointer(VertexAttribLocations.BGCOLOR, 4, gl.FLOAT, false, BYTES_PER_CELL, 13 * Float32Array.BYTES_PER_ELEMENT);
    gl.vertexAttribDivisor(VertexAttribLocations.BGCOLOR, 1);
    gl.enableVertexAttribArray(VertexAttribLocations.CELL_POSITION);
    gl.vertexAttribPointer(VertexAttribLocations.CELL_POSITION, 2, gl.FLOAT, false, BYTES_PER_CELL, 17 * Float32Array.BYTES_PER_ELEMENT);
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
    // One call into the shared core per frame for the whole grid. The
    // frame is wall-clock 60 Hz, which is the rate native and
    // TouchDesigner step at; a render count would run faster whenever the
    // door triggers extra renders.
    this._percell = percellModule();
    const now = performance.now();
    this._percellFrame = Math.floor(now / (1000 / 60));
    // Which code points mark the pixel plane is the shared crate's
    // decision, so the set is read from it rather than written here.
    // Once per module, not once per frame.
    if (this._percell && !this._pixelGlyphs) {
      const marks = new Set<number>();
      for (let i = 0; i < this._percell.pixelGlyphCount(); i++) {
        marks.add(this._percell.pixelGlyph(i));
      }
      this._pixelGlyphs = marks;
    }
    // The shared core decides each cell's layer from its glyph, so the
    // grid crosses before the geometry comes back. The view is taken
    // from the pointer the call returns, since resizing that buffer can
    // grow wasm memory and detach any view made earlier.
    if (this._percell) {
      const cells = this._terminal.cols * this._terminal.rows;
      const glyphs = this._percell.glyphs(cells);
      const buffer = this._terminal.buffer.active;
      for (let y = 0; y < this._terminal.rows; y++) {
        const line = buffer.getLine(buffer.viewportY + y);
        for (let x = 0; x < this._terminal.cols; x++) {
          glyphs[y * this._terminal.cols + x] = line?.getCell(x)?.getCode() ?? 32;
        }
      }
    }
    this._percellGeometry = this._percell?.geometry(this._terminal.cols, this._terminal.rows, this._percellFrame);
    this._substitutionBudget = Math.min(SUBSTITUTION_BUDGET, this._substitutionBudget + (now - this._substitutionRefilledAt) / SUBSTITUTION_REFILL_MS * SUBSTITUTION_BUDGET);
    this._substitutionRefilledAt = now;
    // Cells update in scan order, so a budget spent first come first
    // served would all go to the top rows. Admit roughly one miss in
    // `stride` instead, picked by a hash that moves every frame, so the
    // budget lands across the whole grid.
    this._substitutionStride = Math.max(1, Math.ceil(this._substitutionMisses / SUBSTITUTION_BUDGET));
    this._substitutionMisses = 0;
    return this._atlas ? this._atlas.beginFrame() : true;
  }

  public updateCell(x: number, y: number, code: number, bg: number, fg: number, ext: number, chars: string, width: number, lastBg: number): void {
    // Since this function is called for every cell (`rows*cols`), it must be very optimized. It
    // should not instantiate any variables unless a new glyph is drawn to the cache where the
    // slight slowdown is acceptable for the developer ergonomics provided as it's a once of for
    // each glyph.
    this._updateCell(this._vertices.attributes, x, y, code, bg, fg, ext, chars, width, lastBg);
  }

  /**
   * Whether a cell draws as an opaque pixel-plane quad, background and
   * all. A colour-keyed atlas bakes the colour into the texel and keys
   * the background out to transparent, leaving no coverage mask to mix
   * across, so with tint off nothing takes this path.
   */
  private _isPixelPlane(code: number | undefined): boolean {
    return TextureAtlas.tintGlyphs && code !== undefined && this._pixelGlyphs !== undefined && this._pixelGlyphs.has(code);
  }

  /**
   * The cells the rectangle renderer must not draw a background for,
   * because this renderer already draws them opaque. Undefined means
   * every cell keeps its rectangle, which is the case with tint off.
   */
  public get pixelPlaneGlyphs(): ReadonlySet<number> | undefined {
    return TextureAtlas.tintGlyphs ? this._pixelGlyphs : undefined;
  }

  private _updateCell(array: Float32Array, x: number, y: number, code: number | undefined, bg: number, fg: number, ext: number, chars: string, width: number, lastBg: number): void {
    $i = (y * this._terminal.cols + x) * INDICES_PER_CELL;

    // Exit early if this is a null character, allow space character to continue as it may have
    // underline/strikethrough styles
    if (code === NULL_CELL_CODE || code === undefined/* This is used for the right side of wide chars */) {
      array.fill(0, $i, $i + INDICES_PER_CELL - CELL_POSITION_INDICES);
      return;
    }

    if (!this._atlas) {
      return;
    }

    // Shared per-cell geometry, computed for the whole grid in beginFrame.
    // A stale buffer from before a resize is ignored for that frame.
    const samples = this._terminal.cols * this._terminal.rows;
    const geometry = this._percellGeometry && this._percellGeometry.length === PercellChannel.COUNT * samples ? this._percellGeometry : undefined;
    const sample = y * this._terminal.cols + x;
    if (geometry) {
      if (geometry[PercellChannel.ACTIVE * samples + sample] === 0) {
        array.fill(0, $i, $i + INDICES_PER_CELL - CELL_POSITION_INDICES);
        return;
      }
      // Index modes draw a nearby but wrong glyph. That has to happen
      // before the atlas lookup, so the substitute is what gets drawn.
      const dc = geometry[PercellChannel.DC * samples + sample];
      const dr = geometry[PercellChannel.DR * samples + sample];
      if ((dc !== 0 || dr !== 0) && width === 1 && !(chars && chars.length > 1)) {
        // With tint off the atlas keys glyphs by colour, and the shared core
        // re-rolls substitutions every frame, so unbounded substitution
        // rasterizes new glyphs every frame. Substitutes already in the
        // atlas are free; new ones spend a budget spread across the grid,
        // and a cell that misses out keeps its own glyph this frame. A
        // tinted atlas holds each shape once, so the budget rarely binds.
        const substitute = this._percell!.substitute(code, dc, dr);
        if (this._atlas.hasRasterizedGlyph(substitute, bg, fg, ext)) {
          code = substitute;
        } else {
          this._substitutionMisses++;
          const admitted = ((Math.imul(sample, 0x9e3779b1) ^ Math.imul(this._percellFrame, 0x85ebca6b)) >>> 0) % this._substitutionStride === 0;
          if (admitted && this._substitutionBudget > 0) {
            this._substitutionBudget--;
            code = substitute;
          }
        }
      }
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
        array.fill(0, $i, $i + INDICES_PER_CELL - CELL_POSITION_INDICES);
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
    // a_color: the cell's foreground for a tinted mask, white for a glyph
    // whose colour is already baked into the atlas.
    if (TextureAtlas.tintGlyphs) {
      $fgRgba = this._atlas.getFgColor(bg, fg, ext, chars && chars.length > 1 ? chars.charCodeAt(0) : code);
      array[$i + 9] = (($fgRgba >>> 24) & 0xFF) / 255;
      array[$i + 10] = (($fgRgba >>> 16) & 0xFF) / 255;
      array[$i + 11] = (($fgRgba >>> 8) & 0xFF) / 255;
      array[$i + 12] = ($fgRgba & 0xFF) / 255;
    } else {
      array[$i + 9] = 1;
      array[$i + 10] = 1;
      array[$i + 11] = 1;
      array[$i + 12] = 1;
    }
    // a_bgcolor: a pixel-plane cell draws opaque with its background
    // mixed in and gets no rectangle behind it, so it can move without
    // leaving one. Alpha 0 marks every other cell, which draws as before.
    // A colour-keyed atlas has no coverage mask to mix across, so with
    // tint off these cells stay on the old path, rectangle and all.
    if (this._isPixelPlane(code)) {
      $bgRgba = backgroundRgba(this._themeService, fg, bg);
      array[$i + 13] = (($bgRgba >>> 24) & 0xFF) / 255;
      array[$i + 14] = (($bgRgba >>> 16) & 0xFF) / 255;
      array[$i + 15] = (($bgRgba >>> 8) & 0xFF) / 255;
      array[$i + 16] = 1;
    } else {
      array[$i + 13] = 0;
      array[$i + 14] = 0;
      array[$i + 15] = 0;
      array[$i + 16] = 0;
    }
    // a_cellpos only changes on resize

    // Reduce scale horizontally for wide glyphs printed in cells that would overlap with the
    // following cell (ie. the width is not 2).
    if (this._optionsService.rawOptions.rescaleOverlappingGlyphs) {
      if (allowRescaling(code, width, $glyph.size.x, this._dimensions.device.cell.width)) {
        array[$i + 2] = (this._dimensions.device.cell.width - 1) / this._dimensions.device.canvas.width; // - 1 to improve readability
      }
    }

    if (geometry) {
      // The shared core works in font pixels of an 8x16 cell; scale to
      // this renderer's device cell so one amount looks the same at any
      // font size or pixel ratio.
      array[$i] += geometry[PercellChannel.DX * samples + sample] * this._dimensions.device.cell.width / 8;
      array[$i + 1] += geometry[PercellChannel.DY * samples + sample] * this._dimensions.device.cell.height / 16;
      array[$i + 2] *= geometry[PercellChannel.W_SCALE * samples + sample];
      array[$i + 3] *= geometry[PercellChannel.H_SCALE * samples + sample];
      // Mirror by negating the texture span from the opposite edge.
      if (geometry[PercellChannel.MIRROR_X * samples + sample] !== 0) {
        array[$i + 5] += array[$i + 7];
        array[$i + 7] = -array[$i + 7];
      }
      if (geometry[PercellChannel.MIRROR_Y * samples + sample] !== 0) {
        array[$i + 6] += array[$i + 8];
        array[$i + 8] = -array[$i + 8];
      }
    }

    // page is local to this renderer: only xterm's atlas has pages to draw
    // the wrong one from. It stays out of the shared core.
    if (glyphCfg().badgl === 'page') {
      array[$i + 4] = (array[$i + 4] + 1) % Math.max(1, Math.min(this._atlas.pages.length, this._atlasTextures.length));
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
        this._vertices.attributes[i + 17] = x / terminal.cols;
        this._vertices.attributes[i + 18] = y / terminal.rows;
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
    gl.uniform1f(this._intensityLocation, intensityValue());
    gl.uniform1fv(this._paramLocation, paramValues());
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
