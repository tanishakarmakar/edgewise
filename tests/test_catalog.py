import pytest

from edgewise.catalog import Catalog, estimate_memory


@pytest.fixture(scope="module")
def catalog():
    return Catalog.load()


def test_catalog_loads(catalog):
    assert len(catalog) >= 10
    assert all(v.bits_per_weight > 0 and v.n_layers > 0 for v in catalog)


def test_unknown_model_lists_known_ones(catalog):
    with pytest.raises(KeyError, match="Known models"):
        catalog.get("nope")


def test_memory_estimate_matches_hand_calculation(catalog):
    v = catalog.get("qwen2.5-1.5b-instruct@mlx-4bit")
    est = estimate_memory(v, context_tokens=4096)
    # weights: 1.54e9 params × 4.5 bits / 8 = 0.866 GB
    assert est.weights_gb == pytest.approx(0.86625)
    # KV: 2 × 28 layers × 2 heads × 128 dim × 4096 tokens × 2 bytes = 0.117 GB
    assert est.kv_cache_gb == pytest.approx(0.117440512)
    assert est.total_gb == pytest.approx(0.86625 + 0.117440512 + 0.3)


def test_kv_cache_grows_linearly_with_context(catalog):
    v = catalog.get("llama3.2-3b-instruct@mlx-4bit")
    short, long = estimate_memory(v, 1024), estimate_memory(v, 8192)
    assert long.kv_cache_gb == pytest.approx(8 * short.kv_cache_gb)
    assert long.weights_gb == short.weights_gb


def test_fit_classification(catalog):
    est = estimate_memory(catalog.get("qwen2.5-0.5b-instruct@mlx-4bit"), 4096)
    assert est.fit(budget_gb=10) == "fits"
    assert est.fit(budget_gb=est.total_gb / 0.95) == "tight"
    assert est.fit(budget_gb=est.total_gb - 0.01) == "too big"


def test_8bit_needs_more_memory_than_4bit(catalog):
    q4 = estimate_memory(catalog.get("qwen2.5-3b-instruct@mlx-4bit"))
    q8 = estimate_memory(catalog.get("qwen2.5-3b-instruct@mlx-8bit"))
    assert q8.weights_gb > 1.8 * q4.weights_gb
