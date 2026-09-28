import pandas as pd

from src.data import prepare


def test_toxicity_any_label_mapping_and_stratified_split(tmp_path, monkeypatch):
    raw_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    rows = []
    for i in range(200):
        labels = [0] * 6
        if i % 5 == 0:
            labels[i % 6] = 1  # toxicity expressed through different raw columns
        rows.append(
            {
                "id": i,
                "comment_text": f"comment number {i} text",
                **dict(zip(raw_cols, labels, strict=True)),
            }
        )
    path = tmp_path / "train.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    cfg = prepare.get_config()
    monkeypatch.setitem(cfg["data"]["toxicity"], "max_total", None)
    splits = prepare.build_toxicity({"toxicity": path}, seed=1)

    total = pd.concat(splits.values())
    assert len(total) == 200
    assert total["label"].sum() == 40  # every row with ANY raw label is toxic
    assert set(raw_cols) <= set(total.columns)  # raw labels kept for future extension
    for df in splits.values():
        assert abs(df["label"].mean() - 0.2) < 0.03  # stratified
    again = prepare.build_toxicity({"toxicity": path}, seed=1)
    assert list(again["test"]["text"]) == list(splits["test"]["text"])  # deterministic
