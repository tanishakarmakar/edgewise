"""The key property: accelerated decoding emits exactly what plain greedy decoding emits."""

import pytest

from edgewise.accelerator import (
    CostAwareK,
    DraftProposer,
    FixedK,
    HeuristicK,
    LookupProposer,
    NoProposer,
    generate,
)

EOS = 0
VOCAB = 50


class FakeModel:
    """A deterministic "language model": the next token is a function of the last two tokens.

    `disagree_every` makes it differ from the reference model on some contexts, to play the role
    of an imperfect draft model.
    """

    def __init__(self, disagree_every: int = 0, eos_after: int | None = None):
        self.context: list[int] = []
        self.disagree_every = disagree_every
        self.eos_after = eos_after
        self.forward_calls = 0

    def _next(self, context: list[int]) -> int:
        if self.eos_after is not None and len(context) >= self.eos_after:
            return EOS
        a, b = context[-1], context[-2] if len(context) > 1 else 7
        token = 1 + (a * 31 + b * 17 + len(context) % 3) % (VOCAB - 1)
        if self.disagree_every and (a + b) % self.disagree_every == 0:
            token = 1 + (token % (VOCAB - 1))
        return token

    def forward(self, tokens, last_only=False):
        self.forward_calls += 1
        preds = []
        for token in tokens:
            self.context.append(token)
            preds.append(self._next(self.context))
        if last_only:
            preds = preds[-1:]
        return preds, [-0.1] * len(preds)

    def trim(self, n):
        assert 0 <= n <= len(self.context)
        if n:
            del self.context[-n:]


PROMPT = [3, 9, 4, 12, 5]


def decode(proposer_factory, max_tokens=40, controller=None, eos_after=None):
    target = FakeModel(eos_after=eos_after)
    proposer = proposer_factory()
    steps = list(generate(target, proposer, PROMPT, max_tokens, {EOS}, controller))
    tokens = [t for step in steps for t in step.tokens]
    return tokens, steps, target


def reference(max_tokens=40, eos_after=None):
    return decode(NoProposer, max_tokens, eos_after=eos_after)[0]


def draft(disagree_every):
    return lambda: DraftProposer(FakeModel(disagree_every=disagree_every), PROMPT)


# ------------------------------------------------------------------ losslessness


@pytest.mark.parametrize("disagree_every", [0, 2, 3, 7])
@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_draft_speculation_matches_greedy(disagree_every, k):
    tokens, _, _ = decode(draft(disagree_every), controller=FixedK(k))
    assert tokens == reference()


COSTS = {1: 48.0, 2: 50.0, 5: 88.0, 9: 151.0, 17: 198.0}  # measured on a 3B model, in ms


@pytest.mark.parametrize("disagree_every", [0, 3])
@pytest.mark.parametrize("controller", [HeuristicK, lambda: CostAwareK(COSTS, guess_ms=7.4)])
def test_adaptive_speculation_matches_greedy(disagree_every, controller):
    tokens, _, _ = decode(draft(disagree_every), controller=controller())
    assert tokens == reference()


def test_lookup_matches_greedy():
    tokens, _, _ = decode(lambda: LookupProposer(PROMPT), max_tokens=60, controller=FixedK(8))
    assert tokens == reference(60)


@pytest.mark.parametrize("max_tokens", [1, 2, 5, 13])
def test_never_exceeds_max_tokens(max_tokens):
    tokens, _, _ = decode(draft(0), max_tokens=max_tokens, controller=FixedK(8))
    assert tokens == reference(max_tokens)
    assert len(tokens) == max_tokens


def test_stops_at_eos_and_emits_it_once():
    tokens, _, _ = decode(draft(0), controller=FixedK(8), eos_after=len(PROMPT) + 6)
    assert tokens == reference(eos_after=len(PROMPT) + 6)
    assert tokens[-1] == EOS and tokens.count(EOS) == 1


def test_target_context_is_exactly_the_emitted_sequence():
    tokens, _, target = decode(draft(3), controller=FixedK(4))
    # Everything except the final token has been fed back; no rejected guess is left behind.
    assert target.context == PROMPT + tokens[:-1]


# ------------------------------------------------------------------ it actually saves work


def test_perfect_draft_needs_far_fewer_target_passes():
    _, _, plain = decode(NoProposer)
    _, steps, fast = decode(draft(0), controller=FixedK(4))
    assert plain.forward_calls == 40
    assert fast.forward_calls <= 10
    assert all(s.accepted == s.proposed for s in steps)


def test_first_token_comes_from_the_target_alone():
    _, steps, _ = decode(draft(0))
    assert len(steps[0].tokens) == 1 and steps[0].proposed == 0


# ------------------------------------------------------------------ prompt lookup


def test_lookup_proposes_continuation_of_repeated_ngram():
    proposer = LookupProposer([1, 2, 3, 4, 5, 6, 9, 9, 2, 3])
    assert proposer.propose(3) == [4, 5, 6]


def test_lookup_prefers_most_recent_occurrence_and_longest_ngram():
    proposer = LookupProposer([1, 2, 3, 7, 5, 2, 3, 8, 1, 2, 3])
    assert proposer.propose(1) == [7]  # 3-gram (1,2,3) wins over the more recent 2-gram (2,3)
    proposer = LookupProposer([2, 3, 7, 4, 2, 3, 8, 6, 2, 3])
    assert proposer.propose(1) == [8]  # same 2-gram twice: most recent wins


def test_lookup_proposes_nothing_without_a_match():
    assert LookupProposer([1, 2, 3, 4]).propose(4) == []


def test_lookup_learns_from_generated_tokens():
    proposer = LookupProposer([1, 2])
    proposer.observe([5, 6, 7, 1, 5, 6])
    assert proposer.propose(2) == [7, 1]


# ------------------------------------------------------------------ adaptive k


def test_heuristic_k_grows_on_success_and_shrinks_on_failure():
    ctrl = HeuristicK(start=4, k_max=8)
    ctrl.update(proposed=4, accepted=4)
    assert ctrl.k() == 6
    ctrl.update(proposed=6, accepted=6)
    assert ctrl.k() == 8
    ctrl.update(proposed=8, accepted=8)
    assert ctrl.k() == 8  # capped
    ctrl.update(proposed=8, accepted=5)
    assert ctrl.k() == 7


def test_heuristic_k_pauses_when_guesses_keep_failing_then_probes_again():
    ctrl = HeuristicK(start=4, pause_steps=3)
    for _ in range(10):
        ctrl.update(proposed=ctrl.k() or 1, accepted=0)
        if ctrl._paused:
            break
    assert [ctrl.k() for _ in range(3)] == [0, 0, 0]
    assert ctrl.k() == 1


def test_useless_draft_costs_few_extra_passes_with_adaptive_k():
    class WrongProposer:
        calls = 0

        def propose(self, k):
            WrongProposer.calls += 1
            return [VOCAB + 1] * k  # never matches

        def observe(self, tokens):
            pass

    tokens, _, _ = decode(WrongProposer, controller=HeuristicK())
    assert tokens == reference()
    assert WrongProposer.calls < 20  # paused most of the time


# ------------------------------------------------------------------ cost-aware k


def test_cost_aware_interpolates_verification_cost():
    ctrl = CostAwareK(COSTS)
    assert ctrl._verify_cost(1) == 48
    assert ctrl._verify_cost(3) == pytest.approx(50 + (88 - 50) / 3)
    assert ctrl._verify_cost(20) == pytest.approx(198 + 3 * (198 - 151) / 8)


def test_cost_aware_guesses_more_when_guesses_land():
    confident = CostAwareK(COSTS, guess_ms=7.4, prior=0.95)
    unsure = CostAwareK(COSTS, guess_ms=7.4, prior=0.5)
    assert confident.k() > unsure.k() >= 1


def test_cost_aware_stops_guessing_when_it_does_not_pay():
    # With a costly draft and guesses that rarely land, not guessing is the fastest option.
    ctrl = CostAwareK(COSTS, guess_ms=40.0, prior=0.2, probe_every=4)
    assert [ctrl.k() for _ in range(4)] == [0, 0, 0, 1]  # ...but it still probes occasionally


def test_cost_aware_free_guesses_are_worth_taking_even_when_unsure():
    # Prompt lookup costs nothing to guess; a 2-token pass costs ~4% more than a 1-token pass.
    assert CostAwareK(COSTS, guess_ms=0.0, prior=0.2).k() >= 1


def test_cost_aware_learns_acceptance_probability():
    ctrl = CostAwareK(COSTS, prior=0.5)
    for _ in range(30):
        ctrl.update(proposed=4, accepted=4)
    assert ctrl.p > 0.95
    for _ in range(30):
        ctrl.update(proposed=4, accepted=0)  # only the first guess was really tested
    assert ctrl.p < 0.05


def test_cost_aware_chooses_the_best_rate():
    ctrl = CostAwareK(COSTS, guess_ms=7.4, prior=0.8)
    best = ctrl.k()
    assert all(ctrl.rate(best) >= ctrl.rate(k) for k in range(9))
