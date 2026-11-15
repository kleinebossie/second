/**
 * Common Engine, Lichess State, and Messaging Types for Second Extension.
 */

export interface EngineScore {
  type: 'cp' | 'mate';
  value: number;
}

export interface EnginePv {
  moves: string[];
  score: EngineScore;
}

export interface EngineEval {
  fen: string;
  pvs: EnginePv[];
}

export type Verdict =
  | 'Best'
  | 'Only move'
  | 'Good'
  | 'Inaccuracy'
  | 'Mistake'
  | 'Blunder';

export type PageType = 'analysis' | 'study' | 'game' | 'puzzle' | 'other';

export interface ApiGameData {
  moves: string[];
  whitePlayer?: string;
  blackPlayer?: string;
  opening?: string;
  initialFen?: string;
  variant?: string;
}

export interface LichessGameState {
  url: string;
  pageType: PageType;
  gameId?: string;
  isAllowed: boolean;
  lockReason?: string;
  fen?: string;
  moves?: string[];
  selectedMoveSan?: string;
  selectedMoveUci?: string;
  activePly?: number;
  /**
   * SAN moves of the line currently displayed on the board, from the root
   * position through the selected move. Unlike `moves` (the game's mainline),
   * this can contain moves from a user-explored variation.
   */
  activeLine?: string[];
  /** Lichess tree path of the selected move (e.g. "a1b2"), when available. */
  activePath?: string;
  /** True when the selected move is not part of the game's mainline. */
  activeIsVariation?: boolean;
  whitePlayer?: string;
  blackPlayer?: string;
  opening?: string;
  initialFen?: string;
  variant?: string;
  timestamp: number;
}

export type ExtensionMessage =
  | { type: 'GET_PAGE_STATE' }
  | { type: 'LICHESS_STATE_UPDATE'; payload: LichessGameState }
  | { type: 'PING' };

export interface PingResponse {
  status: 'ok';
}
