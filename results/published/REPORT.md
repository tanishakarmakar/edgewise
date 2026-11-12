# Edgewise results

## Speed and memory

> These runs started with more than 1 GB of swap already in use, so their speed may be understated: llama3.2-3b 4bit, qwen2.5-3b 4bit.

| Model | Prompt tokens | First token (ms) | Tokens/s | Peak memory (MB) |
| --- | ---: | ---: | ---: | ---: |
| qwen2.5-0.5b 4bit | 157 | 97 | 137.9 | 382 |
| qwen2.5-0.5b 4bit | 1053 | 720 | 128.8 | 662 |
| qwen2.5-0.5b 4bit | 4125 | 3321 | 107.8 | 1105 |
| qwen2.5-0.5b 8bit | 157 | 107 | 87.9 | 628 |
| qwen2.5-0.5b 8bit | 1053 | 766 | 83.9 | 908 |
| qwen2.5-0.5b 8bit | 4125 | 3486 | 78.0 | 1335 |
| qwen2.5-1.5b 4bit | 157 | 427 | 45.8 | 1046 |
| qwen2.5-1.5b 4bit | 1053 | 2809 | 44.6 | 1582 |
| qwen2.5-1.5b 4bit | 4125 | 11420 | 43.5 | 1613 |
| qwen2.5-1.5b 8bit | 157 | 467 | 26.7 | 1817 |
| qwen2.5-1.5b 8bit | 1053 | 3290 | 17.0 | 2287 |
| qwen2.5-1.5b 8bit | 4125 | 13231 | 23.4 | 2270 |
| qwen2.5-3b 4bit | 157 | 990 | 21.3 | 1886 |
| qwen2.5-3b 4bit | 1053 | 6254 | 22.3 | 2388 |
| qwen2.5-3b 4bit | 4125 | 25216 | 22.0 | 2518 |
| llama3.2-1b 4bit | 163 | 332 | 61.3 | 850 |
| llama3.2-1b 4bit | 1059 | 2134 | 63.0 | 1385 |
| llama3.2-1b 4bit | 4131 | 8515 | 52.9 | 1501 |
| llama3.2-1b 8bit | 163 | 407 | 35.2 | 1468 |
| llama3.2-1b 8bit | 1059 | 2211 | 34.5 | 1879 |
| llama3.2-1b 8bit | 4131 | 8730 | 31.7 | 2118 |
| llama3.2-3b 4bit | 163 | 1141 | 24.1 | 2014 |
| llama3.2-3b 4bit | 1059 | 7496 | 17.4 | 2456 |
| llama3.2-3b 4bit | 4131 | 24772 | 21.2 | 2835 |

### Memory estimate vs. measured peak

The estimate is the current memory model at the prompt length where the peak was measured. Its working-memory constants were fitted on runs like these, so this shows how well the model fits, not how well it predicts a new device.

| Model | Estimated (GB) | Measured peak (GB) | Error |
| --- | ---: | ---: | ---: |
| qwen2.5-0.5b 4bit | 1.03 | 1.11 | -7% |
| qwen2.5-0.5b 8bit | 1.27 | 1.33 | -5% |
| qwen2.5-1.5b 4bit | 1.68 | 1.61 | +4% |
| qwen2.5-1.5b 8bit | 2.37 | 2.29 | +3% |
| qwen2.5-3b 4bit | 2.59 | 2.52 | +3% |
| llama3.2-1b 4bit | 1.53 | 1.50 | +2% |
| llama3.2-1b 8bit | 2.15 | 2.12 | +2% |
| llama3.2-3b 4bit | 2.98 | 2.84 | +5% |

## Task quality

Mean score per task (0 to 1), standard decoding.

| Model | tool_calling (n=20) | json_extraction (n=16) | grounded_qa (n=16) | short_reasoning (n=20) | text_edit (n=12) |
| --- | ---: | ---: | ---: | ---: | ---: |
| qwen2.5-0.5b 4bit | 0.65 | 0.96 | 0.94 | 0.55 | 1.00 |
| qwen2.5-0.5b 8bit | 0.80 | 0.97 | 0.81 | 0.60 | 1.00 |
| qwen2.5-1.5b 4bit | 0.95 | 0.96 | 0.93 | 0.45 | 1.00 |
| qwen2.5-1.5b 8bit | 0.95 | 0.98 | 0.94 | 0.95 | 1.00 |
| qwen2.5-3b 4bit | 1.00 | 0.99 | 0.94 | 0.75 | 1.00 |
| llama3.2-1b 4bit | 0.60 | 0.63 | 0.91 | 0.50 | 0.97 |
| llama3.2-1b 8bit | 0.70 | 0.38 | 0.91 | 0.80 | 0.95 |
| llama3.2-3b 4bit | 0.90 | 1.00 | 1.00 | 0.90 | 1.00 |

Item counts are small: one item moves a score by 0.05 to 0.08.

![Quality vs. speed for tool_calling](tradeoff-tool_calling.svg)

![Quality vs. speed for json_extraction](tradeoff-json_extraction.svg)

![Quality vs. speed for grounded_qa](tradeoff-grounded_qa.svg)

![Quality vs. speed for short_reasoning](tradeoff-short_reasoning.svg)

![Quality vs. speed for text_edit](tradeoff-text_edit.svg)

## Accelerated decoding

Same model, same prompts. “Decode speed-up” compares tokens per second after the first token; “Whole request” compares total time per request, which also includes reading the prompt (and, for a draft model, the draft reading it too). “Identical outputs” counts items whose text matches standard decoding exactly.

| Model | Decoding | Task | Standard tok/s | Accelerated tok/s | Decode speed-up | Whole request | Guesses accepted | Identical outputs |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen2.5-1.5b 4bit | prompt lookup | grounded_qa | 43.1 | 51.9 | 1.21× | 1.10× | 39% | 16/16 |
| qwen2.5-1.5b 4bit | prompt lookup | json_extraction | 47.2 | 48.9 | 1.03× | 1.04× | 27% | 16/16 |
| qwen2.5-1.5b 4bit | prompt lookup | short_reasoning | 44.0 | 42.7 | 0.97× | 1.06× | 2% | 20/20 |
| qwen2.5-1.5b 4bit | prompt lookup | text_edit | 48.3 | 93.2 | 1.93× | 1.61× | 81% | 12/12 |
| qwen2.5-1.5b 4bit | prompt lookup | tool_calling | 52.3 | 53.7 | 1.03× | 1.07× | 21% | 20/20 |
| qwen2.5-1.5b 4bit | speculative (draft qwen2.5-0.5b 4bit) | grounded_qa | 43.1 | 25.4 | 0.59× | 0.89× | 90% | 16/16 |
| qwen2.5-1.5b 4bit | speculative (draft qwen2.5-0.5b 4bit) | json_extraction | 47.2 | 51.3 | 1.09× | 1.08× | 95% | 16/16 |
| qwen2.5-1.5b 4bit | speculative (draft qwen2.5-0.5b 4bit) | short_reasoning | 44.0 | 29.3 | 0.66× | 0.87× | 49% | 20/20 |
| qwen2.5-1.5b 4bit | speculative (draft qwen2.5-0.5b 4bit) | text_edit | 48.3 | 52.6 | 1.09× | 1.08× | 98% | 12/12 |
| qwen2.5-1.5b 4bit | speculative (draft qwen2.5-0.5b 4bit) | tool_calling | 52.3 | 29.5 | 0.56× | 0.88× | 99% | 20/20 |
| qwen2.5-3b 4bit | prompt lookup | grounded_qa | 26.0 | 32.7 | 1.26× | 1.12× | 50% | 16/16 |
| qwen2.5-3b 4bit | prompt lookup | json_extraction | 23.3 | 27.5 | 1.18× | 1.16× | 26% | 16/16 |
| qwen2.5-3b 4bit | prompt lookup | short_reasoning | 24.5 | 28.0 | 1.14× | 1.13× | 46% | 18/20 |
| qwen2.5-3b 4bit | prompt lookup | text_edit | 25.9 | 43.7 | 1.68× | 1.46× | 67% | 12/12 |
| qwen2.5-3b 4bit | prompt lookup | tool_calling | 24.1 | 27.1 | 1.12× | 1.11× | 20% | 20/20 |
| qwen2.5-3b 4bit | speculative (draft qwen2.5-0.5b 4bit) | grounded_qa | 26.0 | 20.9 | 0.81× | 1.00× | 88% | 16/16 |
| qwen2.5-3b 4bit | speculative (draft qwen2.5-0.5b 4bit) | json_extraction | 23.3 | 34.8 | 1.50× | 1.36× | 97% | 16/16 |
| qwen2.5-3b 4bit | speculative (draft qwen2.5-0.5b 4bit) | short_reasoning | 24.5 | 32.6 | 1.33× | 1.33× | 81% | 16/20 |
| qwen2.5-3b 4bit | speculative (draft qwen2.5-0.5b 4bit) | text_edit | 25.9 | 35.5 | 1.37× | 1.27× | 99% | 12/12 |
| qwen2.5-3b 4bit | speculative (draft qwen2.5-0.5b 4bit) | tool_calling | 24.1 | 21.2 | 0.88× | 0.98× | 96% | 20/20 |

![Decoding speed-up for qwen2.5-3b 4bit](speedup-qwen2.5-3b-instruct-mlx-4bit.svg)

![Decoding speed-up for qwen2.5-1.5b 4bit](speedup-qwen2.5-1.5b-instruct-mlx-4bit.svg)

## Local-first routing

Replayed from stored results on 84 items (grounded_qa, json_extraction, short_reasoning, text_edit, tool_calling): qwen2.5-1.5b 4bit answers first and escalates to qwen2.5-3b 4bit when its confidence is low.

| Policy | Answered on device | Mean score |
| --- | ---: | ---: |
| Local only | 100% | 0.84 |
| Escalate everything | 0% | 0.88 |
| Cascade (escalate below -0.14) | 80% | 0.87 |

The threshold was chosen on 43 items and the table reports the other 41, which played no part in choosing it.

![Cascade quality vs. share answered locally](cascade.svg)
