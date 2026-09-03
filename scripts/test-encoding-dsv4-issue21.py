#!/usr/bin/env python3
"""Unit test for Issue #21 encode_arguments_to_dsml fix (no Docker required)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOTFIX = ROOT / "patches" / "hotfix-encoding-dsv4-issue21.py"


def _load_hotfix():
    import importlib.util

    spec = importlib.util.spec_from_file_location("hotfix_issue21", HOTFIX)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _minimal_encoder(broken: bool) -> str:
    body = (
        """    try:
        arguments = json.loads(tool_call["arguments"])
    except Exception as err:
        arguments = {"arguments": tool_call["arguments"]}"""
        if broken
        else """    raw = tool_call["arguments"]
    if isinstance(raw, dict):
        arguments = raw
    else:
        try:
            arguments = json.loads(raw)
        except (TypeError, ValueError):
            arguments = {"arguments": raw}"""
    )
    return f'''
import json

dsml_token = "｜DSML｜"

def to_json(v):
    return json.dumps(v, ensure_ascii=False)

def encode_arguments_to_dsml(tool_call):
    p_dsml_template = '<{{dsml_token}}parameter name="{{key}}" string="{{is_str}}">{{value}}</{{dsml_token}}parameter>'
    P_dsml_strs = []
{body}
    for k, v in arguments.items():
        p_dsml_str = p_dsml_template.format(
            dsml_token=dsml_token,
            key=k,
            is_str="true" if isinstance(v, str) else "false",
            value=v if isinstance(v, str) else to_json(v),
        )
        P_dsml_strs.append(p_dsml_str)
    return "\\n".join(P_dsml_strs)
'''


class Issue21EncodingTest(unittest.TestCase):
    def test_hotfix_transforms_broken_source(self):
        hotfix = _load_hotfix()
        broken = _minimal_encoder(broken=True)
        updated, status = hotfix.patch_text(broken)
        self.assertEqual(status, "applied")
        self.assertIn("isinstance(raw, dict)", updated)
        again, status2 = hotfix.patch_text(updated)
        self.assertEqual(status2, "skipped")
        self.assertEqual(again, updated)

    def test_dict_and_string_args_match_after_patch(self):
        hotfix = _load_hotfix()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "encoding.py"
            path.write_text(_minimal_encoder(broken=True), encoding="utf-8")
            self.assertEqual(hotfix.patch_file(path), "applied")
            ns: dict = {}
            exec(path.read_text(encoding="utf-8"), ns)
            encode = ns["encode_arguments_to_dsml"]
            as_str = encode({"name": "list_services", "arguments": '{"kind": "airo"}'})
            as_dict = encode({"name": "list_services", "arguments": {"kind": "airo"}})
            self.assertEqual(as_str, as_dict)
            self.assertIn('name="kind"', as_str)
            self.assertNotIn('name="arguments"', as_str)

    def test_broken_dict_path_wraps_arguments(self):
        ns: dict = {}
        exec(_minimal_encoder(broken=True), ns)
        encode = ns["encode_arguments_to_dsml"]
        bad = encode({"name": "list_services", "arguments": {"kind": "airo"}})
        self.assertIn('name="arguments"', bad)

    def test_cli_fails_when_patch_anchor_is_missing(self):
        hotfix = _load_hotfix()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "encoding.py"
            original = "# incompatible encoder\n"
            path.write_text(original, encoding="utf-8")
            self.assertEqual(hotfix.main(["hotfix", str(path)]), 1)
            self.assertEqual(path.read_text(encoding="utf-8"), original)


if __name__ == "__main__":
    unittest.main()
