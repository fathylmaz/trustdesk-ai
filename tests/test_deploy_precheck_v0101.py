from __future__ import annotations
import importlib.util
import json
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / 'scripts' / 'check_deploy_inputs.py'
spec = importlib.util.spec_from_file_location('check_deploy_inputs_v0101', MODULE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def config(n=77, include_num_labels=False):
    obj = {
        'id2label': {str(i): f'bank_intent_{i}' for i in range(n)},
        'label2id': {f'bank_intent_{i}': i for i in range(n)},
    }
    if include_num_labels:
        obj['num_labels'] = n
    return obj


def create_tree(tmp_path, conf):
    for relative in mod.REQUIRED_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'file')
    (tmp_path / mod.CONFIG_REL).write_text(json.dumps(conf), encoding='utf8')


def test_hf_config_without_explicit_num_labels_passes():
    ok, message = mod.inspect_class_mapping(config())
    assert ok
    assert 'etiket eslemesinden' in message


def test_config_with_explicit_num_labels_passes():
    ok, _ = mod.inspect_class_mapping(config(include_num_labels=True))
    assert ok


def test_wrong_class_count_fails():
    ok, _ = mod.inspect_class_mapping(config(76))
    assert not ok


def test_inconsistent_reverse_mapping_fails():
    c = config()
    c['label2id']['bank_intent_0'] = 5
    ok, _ = mod.inspect_class_mapping(c)
    assert not ok


def test_malformed_num_labels_fails():
    c = config()
    c['num_labels'] = 76
    ok, _ = mod.inspect_class_mapping(c)
    assert not ok


def test_full_precheck_passes(tmp_path):
    create_tree(tmp_path, config())
    assert mod.main(tmp_path) == 0


def test_full_precheck_fails_if_file_missing(tmp_path):
    create_tree(tmp_path, config())
    (tmp_path / 'frontend/package.json').unlink()
    assert mod.main(tmp_path) == 1


def test_invalid_json_fails(tmp_path):
    create_tree(tmp_path, config())
    (tmp_path / mod.CONFIG_REL).write_text('{broken', encoding='utf8')
    assert mod.main(tmp_path) == 1
