"""Speculative decoding with greedy verification (see docs/SPEC.md §8).

A cheap *proposer* guesses the next few tokens; the target model checks all of them in one
forward pass and keeps the longest prefix it agrees with. Every emitted token is the target
model's own greedy choice, so the output is the same as decoding one token at a time.

The engine is written against the small `StepModel` interface rather than MLX, so the algorithm
is unit-tested with fake models and reused unchanged on real ones.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol


class StepModel(Protocol):
    """A language model with an internal context (KV cache)."""

    def forward(self, tokens: list[int], last_only: bool = False) -> tuple[list[int], list[float]]:
        """Append `tokens` to the context.

        Returns the greedy next token and its log-probability after each appended token
        (only after the last one if `last_only`, which avoids computing unused outputs for
        long prompts).
        """
        ...

    def trim(self, n: int) -> None:
        """Forget the last `n` tokens of the context."""
        ...


class Proposer(Protocol):
    def propose(self, k: int) -> list[int]:
        """Guess up to `k` tokens that follow everything observed so far."""
        ...

    def observe(self, tokens: list[int]) -> None:
        """Record tokens that were actually emitted."""
        ...


@dataclass
class Step:
    """One verification step: `tokens` were emitted; `accepted` of `proposed` guesses were right."""

    tokens: list[int]
    logprobs: list[float]
    proposed: int = 0
    accepted: int = 0


# ------------------------------------------------------------------ proposers


class NoProposer:
    """Never guesses: turns the engine into plain one-token-at-a-time greedy decoding."""

    def propose(self, k: int) -> list[int]:
        return []

    def observe(self, tokens: list[int]) -> None:
        pass


class DraftProposer:
    """Guesses with a smaller model that shares the target's tokenizer."""

    def __init__(self, draft: StepModel, prompt: list[int]):
        self.draft = draft
        self._pending = list(prompt)  # confirmed tokens the draft hasn't processed yet
        self._speculated: list[int] = []  # guesses currently sitting in the draft's context

    def propose(self, k: int) -> list[int]:
        preds, _ = self.draft.forward(self._pending, last_only=True)
        self._pending = []
        guesses = [preds[-1]]
        while len(guesses) < k:
            preds, _ = self.draft.forward([guesses[-1]])
            guesses.append(preds[-1])
        # The last guess was produced but never fed back into the draft.
        self._speculated = guesses[:-1]
        return guesses

    def observe(self, tokens: list[int]) -> None:
        # Keep the guesses the target agreed with; roll the draft back past the rest.
        valid = 0
        while (
            valid < len(self._speculated)
            and valid < len(tokens)
            and self._speculated[valid] == tokens[valid]
        ):
            valid += 1
        self.draft.trim(len(self._speculated) - valid)
        self._speculated = []
        self._pending.extend(tokens[valid:])


class LookupProposer:
    """Prompt-lookup decoding: if the latest n-gram appeared earlier, guess what followed it.

    Costs no model calls. Works well when the output copies from the input (RAG answers,
    extraction, editing) and does nothing when it doesn't.
    """

    def __init__(self, prompt: list[int], max_ngram: int = 3, min_ngram: int = 2):
        self.sizes = range(max_ngram, min_ngram - 1, -1)
        self.context: list[int] = []
        # n-gram -> position right after its most recent *earlier* occurrence
        self.index: dict[tuple[int, ...], int] = {}
        self.observe(prompt)

    def observe(self, tokens: list[int]) -> None:
        for token in tokens:
            for n in self.sizes:
                if len(self.context) >= n:
                    self.index[tuple(self.context[-n:])] = len(self.context)
            self.context.append(token)

    def propose(self, k: int) -> list[int]:
        for n in self.sizes:
            if len(self.context) < n:
                continue
            position = self.index.get(tuple(self.context[-n:]))
            if position is not None:
                return self.context[position : position + k]
        return []


# ------------------------------------------------------------------ how many tokens to guess


class FixedK:
    def __init__(self, k: int):
        self._k = k

    def k(self) -> int:
        return self._k

    def update(self, proposed: int, accepted: int) -> None:
        pass


class CostAwareK:
    """Pick the guess length that maximizes expected tokens per millisecond.

    Verifying k guesses in one pass is not free: on the hardware this was built on, a pass
    over 9 tokens costs ~3x a pass over 1. Whether guessing pays therefore depends on how
    often guesses are right. With per-token acceptance probability p, a step with k guesses
    emits  1 + p + p^2 + ... + p^k  tokens on average and costs
    verify_ms(k + 1) + k * guess_ms.  This controller estimates p online and chooses the k
    (possibly 0: don't guess) with the best ratio.

    verify_ms: measured cost of one target pass by number of tokens, e.g. {1: 48, 2: 50, 5: 88}
    guess_ms: cost of producing one guess (a draft-model step; 0 for prompt lookup)
    """

    def __init__(
        self,
        verify_ms: dict[int, float],
        guess_ms: float = 0.0,
        k_max: int = 8,
        prior: float = 0.5,
        smoothing: float = 0.2,
        probe_every: int = 8,
    ):
        self.widths = sorted(verify_ms)
        self.verify_ms = verify_ms
        self.guess_ms = guess_ms
        self.k_max = k_max
        self.p = prior
        self.smoothing = smoothing
        self.probe_every = probe_every
        self._since_probe = 0

    def _verify_cost(self, width: int) -> float:
        """Linear interpolation between measured widths (extrapolating from the last two)."""
        ws = self.widths
        if width <= ws[0]:
            return self.verify_ms[ws[0]]
        hi = next((i for i, w in enumerate(ws) if w >= width), len(ws) - 1)
        lo = max(hi - 1, 0)
        if ws[hi] == ws[lo]:
            return self.verify_ms[ws[hi]]
        slope = (self.verify_ms[ws[hi]] - self.verify_ms[ws[lo]]) / (ws[hi] - ws[lo])
        return self.verify_ms[ws[lo]] + slope * (width - ws[lo])

    def rate(self, k: int) -> float:
        """Expected tokens per millisecond when guessing k tokens."""
        expected_tokens = sum(self.p**i for i in range(k + 1))
        return expected_tokens / (self._verify_cost(k + 1) + k * self.guess_ms)

    def k(self) -> int:
        best = max(range(self.k_max + 1), key=self.rate)
        if best > 0:
            return best
        # Not guessing means p never updates, so try a single cheap guess now and then.
        self._since_probe += 1
        if self._since_probe >= self.probe_every:
            self._since_probe = 0
            return 1
        return 0

    def update(self, proposed: int, accepted: int) -> None:
        # Guesses after the first rejection were never really tested.
        tested = accepted + (1 if accepted < proposed else 0)
        for i in range(tested):
            self.p += self.smoothing * ((1.0 if i < accepted else 0.0) - self.p)


class HeuristicK:
    """Guess more when guesses are landing, fewer when they aren't; pause when they never do.

    +2 after a fully accepted step, −1 otherwise (the schedule used by Hugging Face assisted
    generation). If the running acceptance rate stays very low, stop guessing for `pause_steps`
    steps so a bad draft costs nothing, then probe again with a single token. Ignores what a
    verification pass costs; kept as a baseline for CostAwareK.
    """

    def __init__(
        self,
        start: int = 4,
        k_max: int = 8,
        low: float = 0.15,
        pause_steps: int = 8,
        smoothing: float = 0.3,
    ):
        self._k = start
        self.k_max = k_max
        self.low = low
        self.pause_steps = pause_steps
        self.smoothing = smoothing
        self.rate = 0.5
        self._paused = 0

    def k(self) -> int:
        if self._paused > 0:
            self._paused -= 1
            return 0
        return self._k

    def update(self, proposed: int, accepted: int) -> None:
        if proposed == 0:
            return
        self.rate += self.smoothing * (accepted / proposed - self.rate)
        self._k = min(self._k + 2, self.k_max) if accepted == proposed else max(self._k - 1, 1)
        if self.rate < self.low:
            self._paused = self.pause_steps
            self._k = 1
            self.rate = 0.5


# ------------------------------------------------------------------ the decoding loop


def generate(
    target: StepModel,
    proposer: Proposer,
    prompt: list[int],
    max_tokens: int,
    eos_ids: set[int],
    controller: FixedK | HeuristicK | CostAwareK | None = None,
) -> Iterator[Step]:
    """Greedy decoding of `target`, accelerated by `proposer`. Yields one Step per forward pass."""
    if max_tokens <= 0:
        return
    controller = controller or FixedK(4)

    # The first token comes straight from the target, so time-to-first-token is comparable
    # with standard decoding.
    preds, logprobs = target.forward(prompt, last_only=True)
    last = preds[-1]
    proposer.observe([last])
    emitted = 1
    yield Step([last], [logprobs[-1]])

    while emitted < max_tokens and last not in eos_ids:
        # Leave room for the token the target adds itself after the accepted guesses.
        budget = min(controller.k(), max_tokens - emitted - 1)
        guesses = proposer.propose(budget) if budget > 0 else []

        # One pass over [last, g1..gk]: position i predicts the token after guess i.
        preds, logprobs = target.forward([last, *guesses])
        accepted = 0
        while accepted < len(guesses) and guesses[accepted] == preds[accepted]:
            accepted += 1
        tokens = preds[: accepted + 1]  # accepted guesses + the target's own next token

        for i, token in enumerate(tokens):  # stop at the first end-of-sequence token
            if token in eos_ids:
                tokens = tokens[: i + 1]
                break

        # The target processed `last` + every guess; keep `last` + the tokens before the new last.
        target.trim(len(guesses) - (len(tokens) - 1))
        proposer.observe(tokens)
        controller.update(proposed=len(guesses), accepted=accepted)

        last = tokens[-1]
        emitted += len(tokens)
        yield Step(tokens, logprobs[: len(tokens)], proposed=len(guesses), accepted=accepted)
