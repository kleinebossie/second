import { describe, it, expect } from 'vitest';
import { buildPromptForInference } from '../src/core/promptBuilder.js';
import type { EngineEval } from '../src/types.js';

describe('Extension Prompt Builder & Inference Integration', () => {
  const fen = 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1';
  const evalBefore: EngineEval = {
    fen,
    pvs: [
      {
        moves: ['c7c5', 'g1f3'],
        score: { type: 'cp', value: 20 },
      },
    ],
  };

  it('formats prompt with pre-closed reasoning tags for direct answer streaming', () => {
    const res = buildPromptForInference({
      fen,
      moveUci: 'c7c5',
      evalBefore,
      level: 'club',
      detail: 'short',
    });

    expect(res.prompt).toContain('<|im_start|>system');
    expect(res.prompt).toContain('You are Second, a chess assistant.');
    expect(res.prompt).toContain('task: explain_move');
    expect(res.prompt).toContain('level: club');
    expect(res.prompt).toContain('detail: short');
    expect(res.prompt).toContain('Move played: c5.');
    expect(res.prompt).toContain('<|im_start|>assistant\n<think>\n\n</think>\n\n');
    expect(res.prompt.endsWith('<|im_start|>assistant\n<think>\n\n</think>\n\n')).toBe(true);
  });

  it('formats explain_position when no move is supplied', () => {
    const res = buildPromptForInference({
      fen,
      evalBefore,
      level: 'club',
      detail: 'short',
    });

    expect(res.prompt).toContain('task: explain_position');
    expect(res.prompt).not.toContain('Move played:');
    expect(res.prompt).toContain("Black's winning chances:");
  });
});
