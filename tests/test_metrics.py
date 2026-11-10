import pytest

from edgewise.backends.base import TokenStream
from edgewise.bench.metrics import Spread, collect, percentile
from edgewise.schema import TokenEvent

MS = 1_000_000  # ns


def stream(prompt_tokens, times_ms, logprob=-0.5):
    events = (
        TokenEvent(token_id=i, text=f"t{i} ", t_ns=t * MS, logprob=logprob)
        for i, t in enumerate(times_ms)
    )
    return TokenStream(prompt_tokens=prompt_tokens, events=events)


def test_collect_computes_ttft_prefill_and_decode():
    # Request at t=0; first token at 50 ms; then one token every 20 ms.
    result = collect(stream(500, [50, 70, 90, 110, 130]), start_ns=0, peak_memory_mb=123)
    assert result.ttft_ms == pytest.approx(50)
    assert result.prefill_tps == pytest.approx(500 / 0.050)
    assert result.decode_tps == pytest.approx(4 / 0.080)  # 4 tokens after the first, over 80 ms
    assert result.itl_ms == pytest.approx([20, 20, 20, 20])
    assert result.completion_tokens == 5
    assert result.text == "t0 t1 t2 t3 t4 "
    assert result.mean_logprob == pytest.approx(-0.5)
    assert result.peak_memory_mb == 123


def test_collect_single_token_has_no_decode_speed():
    result = collect(stream(10, [30]), start_ns=0)
    assert result.ttft_ms == pytest.approx(30)
    assert result.decode_tps == 0
    assert result.itl_ms == []


def test_collect_empty_stream():
    result = collect(stream(10, []), start_ns=0)
    assert result.completion_tokens == 0
    assert result.text == ""


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([10, 20], 95) == pytest.approx(19.5)
    assert percentile([], 50) == 0


def test_spread_is_robust_to_one_outlier():
    s = Spread.of([10, 10, 11, 10, 100])
    assert s.median == 10
    assert s.n == 5
    assert str(s) == "10.0 [10.0–11.0]"
