"""Delivery 2, acceptance A2: the Exp 4 download script (no network needed)."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fetch_exp4", ROOT / "scripts" / "fetch_exp4.py")
fetch_exp4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch_exp4)


@pytest.fixture
def fake_files(monkeypatch):
    """Replace the pinned files with small fakes and a counting downloader."""
    contents = {name: f"content of {name}".encode() for name in fetch_exp4.FILES}
    files = {name: (rel, hashlib.sha256(contents[name]).hexdigest()) for name, (rel, _) in fetch_exp4.FILES.items()}
    monkeypatch.setattr(fetch_exp4, "FILES", files)
    calls = []

    def download(url):
        calls.append(url)
        rel = url.split("/exp4_fix_car/", 1)[1]
        name = next(n for n, (r, _) in files.items() if r == rel)
        return contents[name]

    return contents, download, calls


def test_pinned_to_a_commit():
    assert len(fetch_exp4.COMMIT) == 40 and fetch_exp4.COMMIT in fetch_exp4.BASE
    assert len(fetch_exp4.FILES) == 5 and all(len(sha) == 64 for _, sha in fetch_exp4.FILES.values())


def test_fetch_then_noop(tmp_path, fake_files):
    contents, download, calls = fake_files
    got = fetch_exp4.fetch(tmp_path, download=download, log=lambda *_: None)
    assert sorted(got) == sorted(contents) and len(calls) == 5
    assert fetch_exp4.missing_or_bad(tmp_path) == []
    assert fetch_exp4.fetch(tmp_path, download=download, log=lambda *_: None) == []
    assert len(calls) == 5  # second run downloaded nothing


def test_tampered_file_fails(tmp_path, fake_files):
    contents, download, calls = fake_files
    fetch_exp4.fetch(tmp_path, download=download, log=lambda *_: None)
    path = tmp_path / "answer.md"
    data = bytearray(path.read_bytes())
    data[0] ^= 1
    path.write_bytes(bytes(data))
    assert fetch_exp4.missing_or_bad(tmp_path) == ["answer.md"]
    with pytest.raises(fetch_exp4.HashMismatch):
        fetch_exp4.fetch(tmp_path, download=download, log=lambda *_: None)
    assert fetch_exp4.fetch(tmp_path, force=True, download=download, log=lambda *_: None) == ["answer.md"]
    assert fetch_exp4.missing_or_bad(tmp_path) == []


def test_bad_download_is_not_written(tmp_path, fake_files):
    _, _, _ = fake_files
    with pytest.raises(fetch_exp4.HashMismatch):
        fetch_exp4.fetch(tmp_path, download=lambda url: b"not the file", log=lambda *_: None)
    assert not any(tmp_path.iterdir())


def test_check_mode(tmp_path, fake_files, capsys):
    assert fetch_exp4.main(["--check", "--dest", str(tmp_path)]) == 1
