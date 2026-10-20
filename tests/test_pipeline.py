from pathlib import Path
import pandas as pd
from trustdesk.data import load_csv
from trustdesk.inference import classify, load_local_model
from trustdesk.train import train_and_evaluate


def make_fake_data(tmp_path: Path):
    # These examples test code paths ONLY. They are NOT real BANKING77 data.
    templates = {
        "card_stolen": ["my card was stolen", "someone took my card", "my missing credit card", "stolen payment card"],
        "change_pin": ["change my pin number", "update card pin", "new pin code", "forgot pin number"],
        "refund": ["where is my refund", "refund not received", "refund still pending", "missing refund money"],
    }
    train = [{"text": f"{text} please {i}", "category": name}
             for name, examples in templates.items() for text in examples for i in range(3)]
    test = [{"text": examples[0], "category": name} for name, examples in templates.items()]
    train_path, test_path = tmp_path / "train.csv", tmp_path / "test.csv"
    pd.DataFrame(train).to_csv(train_path, index=False)
    pd.DataFrame(test).to_csv(test_path, index=False)
    return train_path, test_path


def test_training_and_inference(tmp_path):
    train_path, test_path = make_fake_data(tmp_path)
    assert len(load_csv(train_path)) == 36
    report = train_and_evaluate(train_path, test_path, tmp_path / "out")
    assert report["class_count"] == 3
    assert 0.0 <= report["validation"]["macro_f1"] <= 1.0
    model = load_local_model(tmp_path / "out" / "baseline.joblib")
    output = classify(model, "my payment card was stolen")
    assert len(output["top_predictions"]) == 3
    assert 0 <= output["score"] <= 1
    assert isinstance(output["needs_human_review"], bool)
