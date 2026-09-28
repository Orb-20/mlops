from src.training.metrics import scalar_metrics, sentiment_metrics, toxicity_metrics


def test_sentiment_perfect():
    m = sentiment_metrics([0, 1, 2, 2], [0, 1, 2, 2])
    assert m["sentiment_macro_f1"] == 1.0
    assert m["sentiment_confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 0, 2]]


def test_toxicity_threshold_and_auc():
    m = toxicity_metrics([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9], threshold=0.5)
    assert m["toxicity_precision"] == 0.5 and m["toxicity_recall"] == 0.5
    assert m["toxicity_roc_auc"] == 0.75
    assert "toxicity_pr_auc" in m


def test_toxicity_single_class_has_no_auc():
    m = toxicity_metrics([0, 0], [0.1, 0.2])
    assert "toxicity_roc_auc" not in m


def test_scalar_metrics_drops_matrices():
    s = scalar_metrics(sentiment_metrics([0, 1], [0, 1]))
    assert "sentiment_confusion_matrix" not in s and "sentiment_macro_f1" in s
