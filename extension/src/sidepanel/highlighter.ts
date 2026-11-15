/**
 * Semantic Chess Token Highlighter for Second Coach Commentary.
 *
 * Highlights strictly three token categories in the coach commentary stream:
 * 1. Moves (`tok-move`): Warm ochre (#e5c07b) monospace with fine dashed underline.
 *    Handles numbered SAN moves (e.g., `24... Nd4`, `24… Nd4`, `25. Bxd5!`, `1. e4`),
 *    castling (`O-O`, `O-O-O`, `0-0`, `0-0-0` with optional `+#!?`),
 *    long algebraic (`Qd1-h5`, `e2-e4`), and standalone SAN tokens.
 * 2. Squares (`tok-square`): Slate chalk (#cbd5e1) monospace with pronounced 1.5px solid underline.
 *    Matches algebraic board coordinates (`a1` through `h8`).
 * 3. Pieces (`tok-piece`): Crisp semibold white (#ffffff) for chess piece names
 *    (`king`, `queen`, `rook`, `bishop`, `knight`, `pawn`, and plurals), excluding
 *    opening names like "King's Indian" or "Queen's Gambit".
 *
 * Also provides pure helper functions for Variation 6A header formatting.
 */

export type TokenKind = 'text' | 'move' | 'square' | 'piece';

export interface CommentaryToken {
  kind: TokenKind;
  value: string;
}

/**
 * Escapes raw text for safe HTML insertion.
 */
export function escapeHtml(str: string): string {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/**
 * Formats a move with its ply number (e.g. `24... Nd4` or `25. Bxd5`) when ply is known.
 */
export function formatMoveWithNumber(moveSan?: string, activePly?: number): string {
  if (!moveSan || !moveSan.trim()) return '--';
  const cleaned = moveSan.trim().replace(/\u2026/g, '...');
  if (/^\d+\./.test(cleaned)) {
    return cleaned;
  }
  if (activePly && activePly > 0) {
    const moveNumber = Math.ceil(activePly / 2);
    const isWhite = activePly % 2 === 1;
    return `${moveNumber}${isWhite ? '.' : '...'} ${cleaned}`;
  }
  return cleaned;
}

// White move number prefix (e.g. `24.`, `1. `) vs Black ellipsis prefix (`24...`, `24…`)
const WHITE_MOVE_NUM_PREFIX = '(?:\\d+\\.(?!\\.)\\s*)';
const BLACK_MOVE_NUM_PREFIX = '(?:\\d+(?:\\.\\.\\.|\\u2026)\\s*)';

// Castling (letter O or digit 0) with optional check/mate/annotation suffixes
const CASTLE_BODY = '(?:O-O(?:-O)?|0-0(?:-0)?)[+#!?]*';

// Piece SAN or Long Algebraic move: e.g. `Nd4`, `Bxd5!`, `Qd1-h5`, `Qd1–h5`, `Rae8+`, `Nbd7`
const PIECE_MOVE_BODY = '[KQRBN][a-h]?[1-8]?(?:[-–—x]?[a-h][1-8])(?:=[QRBN])?[+#!?]*';

// Unambiguous pawn move:
// - capture (`exd5`, `e2xe4`)
// - same-file long algebraic push with valid 1-2 rank pawn step (`e2-e4`, `e2–e4`, `a7-a8=Q`)
//   (avoids false positives on diagonals like `a1-h8` or full-file spans like `a1-a8`)
// - promotion (`e8=Q+`) or check/annotated (`e4+`, `d5!`)
const PAWN_LONG_PUSH = `(?:${['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
  .map(
    (f) =>
      `${f}(?:2[-–—]${f}[134]|3[-–—]${f}[24]|4[-–—]${f}[35]|5[-–—]${f}[46]|6[-–—]${f}[57]|7[-–—]${f}[568])`
  )
  .join('|')})(?:=[QRBN])?[+#!?]*`;
const UNAMBIGUOUS_PAWN_BODY =
  `(?:[a-h][1-8]?x[a-h][1-8](?:=[QRBN])?[+#!?]*|${PAWN_LONG_PUSH}|[a-h][1-8]=[QRBN][+#!?]*|[a-h][1-8][+#!?]+)`;

// Simple pawn push when preceded by a move number (`1. e4`, `24... d5`)
const NUMBERED_PAWN_BODY = '[a-h][1-8](?:=[QRBN])?[+#!?]*';

// Single move body (for White or Black numbered moves)
const ANY_MOVE_BODY = `(?:${CASTLE_BODY}|${PIECE_MOVE_BODY}|${UNAMBIGUOUS_PAWN_BODY}|${NUMBERED_PAWN_BODY})`;

// Second half of a White+Black move pair (`1. e4 e5`):
// Always allow unambiguous replies (castling, piece moves, captures, promotions, checks/annotations).
// Only treat a bare coordinate as Black's pawn reply when it is on ranks 2-6 (Black pawns cannot
// quietly advance to ranks 1, 7, or 8 without capture/promotion) and is not followed by prose words
// indicating a square reference (e.g. "After 24. Nd4 f7 is weak" or "After 1. e4\nd5 is weak").
const SQUARE_PROSE_FOLLOW =
  '(?:is|was|are|were|becomes?|remains?|looks?|seems?|gets?|stays?|falls?|stands?|sits?|lies?|square|squares|pawn|pawns|outpost|hole|weakness|diagonal|file|rank|and\\s+[a-h][1-8]|or\\s+[a-h][1-8]|on|at|for|with|to|from)';
const BLACK_BARE_PAWN_REPLY = `[a-h][2-6](?!\\s+${SQUARE_PROSE_FOLLOW}\\b)`;
const BLACK_REPLY_BODY = `(?:${CASTLE_BODY}|${PIECE_MOVE_BODY}|${UNAMBIGUOUS_PAWN_BODY}|${BLACK_BARE_PAWN_REPLY})`;

const PIECE_WORD = '(?:[Kk]ings?|[Qq]ueens?|[Rr]ooks?|[Bb]ishops?|[Kk]nights?|[Pp]awns?)';
const CAPITALIZED_PIECE_WORD = '(?:Kings?|Queens?|Rooks?|Bishops?|Knights?|Pawns?)';
const OPENING_CORE_NOUN = '(?:Indian|Gambit|Opening|Game|System|Countergambit|Counter-?gambit)';
const OPENING_TRAIL = '(?:Indian|Gambit|Defense|Defence|Attack|Opening|Game|Variation|System|Accepted|Declined|Countergambit|Counter-?gambit)';
const OPENING_PHRASE =
  `${CAPITALIZED_PIECE_WORD}(?:\\s*['\\u2019]s)?\\s+(?:${OPENING_CORE_NOUN}|${CAPITALIZED_PIECE_WORD}(?:\\s*['\\u2019]s)?\\s+${OPENING_TRAIL})(?:\\s+${OPENING_TRAIL})*`;

const TOKEN_REGEX = new RegExp(
  '(?<![A-Za-z0-9_])(?:' +
    // Group: Chess Moves (White move with optional Black reply, Black ellipsis single move, or unnumbered move)
    '(?<move>' +
      WHITE_MOVE_NUM_PREFIX +
      ANY_MOVE_BODY +
      '(?:\\s+' + BLACK_REPLY_BODY + ')?' +
      '|' +
      BLACK_MOVE_NUM_PREFIX +
      ANY_MOVE_BODY +
      '|' +
      '(?:' +
        CASTLE_BODY +
        '|' +
        PIECE_MOVE_BODY +
        '|' +
        UNAMBIGUOUS_PAWN_BODY +
      ')' +
    ')(?![A-Za-z0-9_])' +
    '|' +
    // Group: Board Squares (e.g. f7, a8, d4; allows f7-pawn, f7-square, a1-h8)
    '(?<square>[a-h][1-8])(?![A-Za-z0-9_+#!?=])' +
    '|' +
    // Group: Capitalized opening names consumed as plain text so trailing piece words don't leak
    `(?<opening>${OPENING_PHRASE})(?![A-Za-z0-9_])` +
    '|' +
    // Group: Chess Pieces
    `(?<piece>${PIECE_WORD})(?![A-Za-z0-9_])` +
  ')',
  'g'
);

const WHITE_PAIR_SPLIT_REGEX = /^(\d+\.(?!\.)\s*\S+)(\s+)(\S+)$/;

/**
 * Strips simple inline Markdown delimiters (`**bold**`, `__bold__`, `` `code` ``)
 * that small local models occasionally emit, keeping commentary strictly unboxed.
 */
export function stripInlineMarkdown(text: string): string {
  return text
    .replace(/`([^`\n]+)`/g, '$1')
    .replace(/\*\*([^*\n]+)\*\*/g, '$1')
    .replace(/__([^_\n]+)__/g, '$1');
}

function tokenizeCleanCommentary(cleanText: string): CommentaryToken[] {
  if (!cleanText) return [];

  const tokens: CommentaryToken[] = [];
  let lastIndex = 0;

  TOKEN_REGEX.lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = TOKEN_REGEX.exec(cleanText)) !== null) {
    const matchIndex = match.index;
    if (matchIndex > lastIndex) {
      tokens.push({
        kind: 'text',
        value: cleanText.slice(lastIndex, matchIndex),
      });
    }

    const groups = match.groups;
    if (groups?.move !== undefined) {
      const moveMatch = groups.move;
      // Split White move pairs (`1. e4 e5`, `12. 0-0 Nf6`, `1. e4\ne5`) while preserving whitespace
      const pairMatch = moveMatch.match(WHITE_PAIR_SPLIT_REGEX);
      if (pairMatch) {
        tokens.push({ kind: 'move', value: pairMatch[1] });
        tokens.push({ kind: 'text', value: pairMatch[2] });
        tokens.push({ kind: 'move', value: pairMatch[3] });
      } else {
        tokens.push({ kind: 'move', value: moveMatch });
      }
    } else if (groups?.square !== undefined) {
      tokens.push({ kind: 'square', value: groups.square });
    } else if (groups?.opening !== undefined) {
      tokens.push({ kind: 'text', value: groups.opening });
    } else if (groups?.piece !== undefined) {
      tokens.push({ kind: 'piece', value: groups.piece });
    }

    lastIndex = TOKEN_REGEX.lastIndex;
  }

  if (lastIndex < cleanText.length) {
    tokens.push({
      kind: 'text',
      value: cleanText.slice(lastIndex),
    });
  }

  return tokens;
}

/**
 * Tokenizes raw commentary text into structured tokens for testing and rendering.
 */
export function tokenizeCommentary(text: string): CommentaryToken[] {
  if (!text) return [];
  return tokenizeCleanCommentary(stripInlineMarkdown(text));
}

function renderTokensToHtml(tokens: CommentaryToken[]): string {
  return tokens
    .map((tok) => {
      const escaped = escapeHtml(tok.value).replace(/\n/g, '<br>');
      switch (tok.kind) {
        case 'move':
          return `<span class="tok-move" data-move="${escapeHtml(tok.value)}">${escaped}</span>`;
        case 'square':
          return `<span class="tok-square" data-square="${escapeHtml(tok.value)}">${escaped}</span>`;
        case 'piece':
          return `<span class="tok-piece">${escaped}</span>`;
        default:
          return escaped;
      }
    })
    .join('');
}

/**
 * Formats a single commentary paragraph into highlighted inner HTML for incremental streaming.
 */
export function formatParagraphInnerHtml(rawParagraph: string): string {
  const cleaned = stripInlineMarkdown(rawParagraph).replace(/\r\n/g, '\n').trim();
  if (!cleaned) return '';
  return renderTokensToHtml(tokenizeCleanCommentary(cleaned));
}

/**
 * Formats raw model commentary into safe HTML with semantic `<span class="tok-*">` wrappers
 * and clean paragraph spacing.
 */
export function formatCommentaryHtml(text: string): string {
  if (!text || !text.trim()) return '';

  const normalized = stripInlineMarkdown(text).replace(/\r\n/g, '\n');
  const paragraphs = normalized
    .split(/\n{2,}/)
    .map((p) => p.trim())
    .filter((p) => p.length > 0);

  return paragraphs
    .map((para) => `<p class="commentary-para">${renderTokensToHtml(tokenizeCleanCommentary(para))}</p>`)
    .join('');
}
