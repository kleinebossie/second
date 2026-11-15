import { describe, it, expect } from 'vitest';
import {
  tokenizeCommentary,
  formatCommentaryHtml,
  formatParagraphInnerHtml,
  formatMoveWithNumber,
} from '../src/sidepanel/highlighter.js';

describe('Semantic Chess Token Highlighter', () => {
  it('highlights numbered moves, unnumbered moves, squares, and pieces accurately', () => {
    const sample =
      'Moving the knight to 24... Nd4 is a critical tactical mistake that surrenders the initiative.\n\n' +
      'While jumping into the center appears active, it abandons the defense of the f7 square. ' +
      'White refutes this immediately with 25. Bxd5!, unleashing an unstoppable double attack against ' +
      'the black king and the undefended rook on a8.';

    const tokens = tokenizeCommentary(sample);
    const moves = tokens.filter((t) => t.kind === 'move').map((t) => t.value);
    const squares = tokens.filter((t) => t.kind === 'square').map((t) => t.value);
    const pieces = tokens.filter((t) => t.kind === 'piece').map((t) => t.value);

    expect(moves).toEqual(['24... Nd4', '25. Bxd5!']);
    expect(squares).toEqual(['f7', 'a8']);
    expect(pieces).toEqual(['knight', 'king', 'rook']);
  });

  it('distinguishes numbered pawn advances and move pairs from standalone square references', () => {
    const sample = 'After 1. e4 e5, control of e4 and d5 is contested by both pawns.';
    const tokens = tokenizeCommentary(sample);

    const moves = tokens.filter((t) => t.kind === 'move').map((t) => t.value);
    const squares = tokens.filter((t) => t.kind === 'square').map((t) => t.value);
    const pieces = tokens.filter((t) => t.kind === 'piece').map((t) => t.value);

    expect(moves).toEqual(['1. e4', 'e5']);
    expect(squares).toEqual(['e4', 'd5']);
    expect(pieces).toEqual(['pawns']);
  });

  it('highlights castling (letter O and digit 0) with check/mate suffixes, promotions, and checks', () => {
    const sample =
      'Black castles with O-O-O#, or 12... O-O+, or 0-0, while White plays exd5, e2xe4, e8=Q+, and Qxf7#.';
    const tokens = tokenizeCommentary(sample);
    const moves = tokens.filter((t) => t.kind === 'move').map((t) => t.value);

    expect(moves).toEqual(['O-O-O#', '12... O-O+', '0-0', 'exd5', 'e2xe4', 'e8=Q+', 'Qxf7#']);
  });

  it('handles Unicode ellipsis, en-dash in long algebraic, and hyphenated board squares', () => {
    const sample = 'After 24… Nd4, White plays Qd1–h5+ or e2–e4 to target the f7-square and f7-pawn.';
    const tokens = tokenizeCommentary(sample);
    const moves = tokens.filter((t) => t.kind === 'move').map((t) => t.value);
    const squares = tokens.filter((t) => t.kind === 'square').map((t) => t.value);

    expect(moves).toEqual(['24… Nd4', 'Qd1–h5+', 'e2–e4']);
    expect(squares).toEqual(['f7', 'f7']);
  });

  it('does not falsely highlight opening names like King’s Indian, Kings Indian, or Queen’s Gambit as pieces', () => {
    const sample =
      "In the King's Indian Defense, Kings Indian, and Queen’s Gambit, the white king and black queen remain safe, while Nf3abc is untouched.";
    const tokens = tokenizeCommentary(sample);
    const pieces = tokens.filter((t) => t.kind === 'piece').map((t) => t.value);
    const moves = tokens.filter((t) => t.kind === 'move').map((t) => t.value);

    expect(pieces).toEqual(['king', 'queen']);
    expect(moves).toEqual([]);
  });

  it('does not highlight tactical motif phrases or file names', () => {
    const sample = 'This tactical mistake allows a double attack along the c-file.';
    const html = formatCommentaryHtml(sample);

    expect(html).not.toContain('tok-move');
    expect(html).not.toContain('tok-square');
    expect(html).not.toContain('tok-piece');
    expect(html).toBe(
      '<p class="commentary-para">This tactical mistake allows a double attack along the c-file.</p>'
    );
  });

  it('escapes HTML entities to prevent XSS injection and handles empty/whitespace inputs', () => {
    const malicious = '<script>alert("xss")</script> on f7 with 24... Nd4';
    const html = formatCommentaryHtml(malicious);

    expect(html).not.toContain('<script>');
    expect(html).toContain('&lt;script&gt;');
    expect(html).toContain('<span class="tok-square" data-square="f7">f7</span>');
    expect(html).toContain('<span class="tok-move" data-move="24... Nd4">24... Nd4</span>');

    expect(formatCommentaryHtml('')).toBe('');
    expect(formatCommentaryHtml('   \n  ')).toBe('');
  });

  it('formats moves with ply numbers accurately in formatMoveWithNumber', () => {
    expect(formatMoveWithNumber(undefined, 10)).toBe('--');
    expect(formatMoveWithNumber('   ', 10)).toBe('--');
    expect(formatMoveWithNumber('e4', 1)).toBe('1. e4');
    expect(formatMoveWithNumber('e5', 2)).toBe('1... e5');
    expect(formatMoveWithNumber('Nd4', 48)).toBe('24... Nd4');
    expect(formatMoveWithNumber('Bxd5!', 49)).toBe('25. Bxd5!');
    expect(formatMoveWithNumber('24… Nd4', 48)).toBe('24... Nd4');
    expect(formatMoveWithNumber('Nf3')).toBe('Nf3');
  });

  it('does not misclassify diagonals or full-file spans like a1-h8, h1-a8, or a1-a8 as moves', () => {
    const sample = 'Control of the a1-h8 diagonal, h1-a8, and the a1-a8 file decides the game.';
    const tokens = tokenizeCommentary(sample);
    const moves = tokens.filter((t) => t.kind === 'move').map((t) => t.value);
    const squares = tokens.filter((t) => t.kind === 'square').map((t) => t.value);

    expect(moves).toEqual([]);
    expect(squares).toEqual(['a1', 'h8', 'h1', 'a8', 'a1', 'a8']);
  });

  it('does not merge a Black ellipsis move with a following square into a move pair', () => {
    const sample = 'After 24... Nd4 e4 is weak and vulnerable.';
    const tokens = tokenizeCommentary(sample);
    const moves = tokens.filter((t) => t.kind === 'move').map((t) => t.value);
    const squares = tokens.filter((t) => t.kind === 'square').map((t) => t.value);

    expect(moves).toEqual(['24... Nd4']);
    expect(squares).toEqual(['e4']);
  });

  it('splits digit-castling move pairs and preserves newlines between paired moves', () => {
    const tokens = tokenizeCommentary('12. 0-0 Nf6 is solid.');
    const moves = tokens.filter((t) => t.kind === 'move').map((t) => t.value);
    expect(moves).toEqual(['12. 0-0', 'Nf6']);

    const htmlWithNewline = formatCommentaryHtml('1. e4\ne5');
    expect(htmlWithNewline).toBe(
      '<p class="commentary-para"><span class="tok-move" data-move="1. e4">1. e4</span><br><span class="tok-move" data-move="e5">e5</span></p>'
    );
  });

  it('excludes multi-word opening names without leaking trailing piece names', () => {
    const sample =
      "In the King's Knight Opening, King's Pawn Game, and Queen's Gambit Declined, the king's pawn on f7 is critical.";
    const tokens = tokenizeCommentary(sample);
    const pieces = tokens.filter((t) => t.kind === 'piece').map((t) => t.value);
    const squares = tokens.filter((t) => t.kind === 'square').map((t) => t.value);

    expect(pieces).toEqual(['king', 'pawn']);
    expect(squares).toEqual(['f7']);
  });

  it('does not hijack following standalone squares after a White move on spaces or newlines', () => {
    const proseSpace = tokenizeCommentary('After 24. Nd4 f7 is weak.');
    expect(proseSpace.filter((t) => t.kind === 'move').map((t) => t.value)).toEqual(['24. Nd4']);
    expect(proseSpace.filter((t) => t.kind === 'square').map((t) => t.value)).toEqual(['f7']);

    const proseNewline = tokenizeCommentary('After 1. e4\nd5 is weak.');
    expect(proseNewline.filter((t) => t.kind === 'move').map((t) => t.value)).toEqual(['1. e4']);
    expect(proseNewline.filter((t) => t.kind === 'square').map((t) => t.value)).toEqual(['d5']);
  });

  it('preserves piece highlights in prose phrases like The King’s Attack and strips inline Markdown/blank paragraphs', () => {
    const prose = tokenizeCommentary("The King's Attack and White's Queen's Defense crumble.");
    expect(prose.filter((t) => t.kind === 'piece').map((t) => t.value)).toEqual(['King', 'Queen']);

    const html = formatCommentaryHtml('\n\nPlay **24... Nd4** to target `f7`.\n\n');
    expect(html).toBe(
      '<p class="commentary-para">Play <span class="tok-move" data-move="24... Nd4">24... Nd4</span> to target <span class="tok-square" data-square="f7">f7</span>.</p>'
    );

    const pushes = tokenizeCommentary('Compare a2-a4 and h7-h5 against the a1-h8 diagonal.');
    expect(pushes.filter((t) => t.kind === 'move').map((t) => t.value)).toEqual(['a2-a4', 'h7-h5']);
    expect(pushes.filter((t) => t.kind === 'square').map((t) => t.value)).toEqual(['a1', 'h8']);
  });

  it('formats a single paragraph inner HTML cleanly for incremental streaming', () => {
    const inner = formatParagraphInnerHtml('Play **24... Nd4** targeting `f7`.');
    expect(inner).toBe(
      'Play <span class="tok-move" data-move="24... Nd4">24... Nd4</span> targeting <span class="tok-square" data-square="f7">f7</span>.'
    );
  });
});
