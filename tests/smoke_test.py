from vocab_estimator.coverage import DocumentCoverageAnalyzer
from vocab_estimator.sampler import VocabularySampler
from vocab_estimator.vocab_bank import VocabBank
from vocab_estimator.vocab_model import VocabEstimator


def test_smoke():
    bank = VocabBank()
    assert len(bank) > 0
    assert bank.get_rank("running") == bank.get_rank("run") or bank.get_rank("running") is None

    sampler = VocabularySampler(bank)
    test_items = sampler.generate_test_list(per_bucket=2)
    assert test_items

    responses = [(word, idx % 2 == 0) for idx, (word, _, _) in enumerate(test_items[:20])]
    result = VocabEstimator(bank).estimate_single(responses)
    assert result["point_estimate"] >= 0
    assert result["confidence"] in {"高", "中", "低"}

    coverage = DocumentCoverageAnalyzer(bank).analyze_text("Students read books and analyze evidence.")
    assert coverage["token_count"] > 0
