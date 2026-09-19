/**
 * Copyright (c) 2026 The xterm.js authors. All rights reserved.
 * @license MIT
 */

import { IDisposable, Terminal } from '@xterm/xterm';
import { applyPayload, hashLine, PlaneRow } from './PlanePayload';

/** The private OSC ids the door uses: a row's hidden cells, and the start of a row. */
const OSC_PLANE = 7770;
const OSC_ROW_START = 7771;

/**
 * The plane payloads the door has sent, kept per row beside the terminal's own
 * buffer. The flat cells go through xterm as ever, and an OSC it would ignore
 * reaches these handlers instead, so the core is untouched.
 *
 * Every row of a frame with planes opens with `ESC ] 7771 ; shift BEL` (zero
 * when the row is not banded). It arrives with the cursor on the row about to
 * be painted, so it is where the row's old payloads are dropped and its shift
 * is taken. The payloads of the paint, one per plane, follow the row's cells
 * with the cursor still on the row. A row painted without the marker keeps its
 * payloads only while its flat cells are the ones they arrived with.
 */
export class PlaneStore implements IDisposable {
  private readonly _rows = { normal: new Map<number, PlaneRow>(), alternate: new Map<number, PlaneRow>() };
  private readonly _disposables: IDisposable[] = [];

  constructor(private readonly _terminal: Terminal) {
    this._disposables.push(
      _terminal.parser.registerOscHandler(OSC_PLANE, data => this._plane(data)),
      _terminal.parser.registerOscHandler(OSC_ROW_START, data => this._rowStart(data))
    );
  }

  public dispose(): void {
    for (const disposable of this._disposables) {
      disposable.dispose();
    }
    this._rows.normal.clear();
    this._rows.alternate.clear();
  }

  /** Whether any row of the active buffer holds plane data. */
  public get hasRows(): boolean {
    return this._map().size > 0;
  }

  /** The plane data for an absolute buffer row, or undefined. */
  public rowAt(absoluteRow: number): PlaneRow | undefined {
    return this._map().get(absoluteRow);
  }

  private _map(): Map<number, PlaneRow> {
    return this._terminal.buffer.active.type === 'alternate' ? this._rows.alternate : this._rows.normal;
  }

  private _row(create: boolean): PlaneRow | undefined {
    const buffer = this._terminal.buffer.active;
    const absoluteRow = buffer.baseY + buffer.cursorY;
    const map = this._map();
    let row = map.get(absoluteRow);
    if (!row && create) {
      row = new PlaneRow();
      map.set(absoluteRow, row);
      this._prune(map);
    }
    return row;
  }

  private _rowStart(data: string): boolean {
    const shift = parseInt(data, 10);
    const row = this._row(true)!;
    row.reset();
    row.shift = Number.isFinite(shift) && shift > 0 ? shift : 0;
    return true;
  }

  private _plane(data: string): boolean {
    const separator = data.indexOf(';');
    const tag = parseInt(data.slice(0, separator), 10);
    if (separator < 0 || !(tag >= 1 && tag <= 7)) {
      return true;
    }
    const buffer = this._terminal.buffer.active;
    const line = (this._terminal as any)._core.buffer.lines.get(buffer.baseY + buffer.cursorY);
    if (!line) {
      return true;
    }
    const row = this._row(true)!;
    if (applyPayload(row, tag, data.slice(separator + 1), this._terminal.cols)) {
      // The row's cells are complete once its payloads have arrived, so the
      // last payload's snapshot is the one to compare against.
      row.hash = hashLine(line, this._terminal.cols);
    }
    return true;
  }

  /** Drops rows well above the viewport, so scrollback does not grow the map for ever. */
  private _prune(map: Map<number, PlaneRow>): void {
    const rows = this._terminal.rows;
    if (map.size <= rows * 4) {
      return;
    }
    const floor = this._terminal.buffer.active.viewportY - rows;
    for (const key of map.keys()) {
      if (key < floor) {
        map.delete(key);
      }
    }
  }
}
