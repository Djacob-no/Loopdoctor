"""Acceptance I1 (human play CLI), I3 (demo CLI) and the H2 benchmark script."""

from __future__ import annotations

import importlib.util
import io
import subprocess
import sys
from pathlib import Path

import pytest

from weightshift.cli import demo, play
from weightshift.env import Game, load_fixed_layout
from weightshift.env.constants import DOWN, DROP, LEFT, NOOP, PICK, RIGHT, UP
from weightshift.trajectory import Trajectory, verify_replay

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ I1 keys
@pytest.mark.parametrize(
    "key,action",
    [
        ("w", UP), ("a", LEFT), ("s", DOWN), ("d", RIGHT),
        ("W", UP), ("\x1b[A", UP), ("\x1b[B", DOWN), ("\x1b[C", RIGHT), ("\x1b[D", LEFT),
        ("p", PICK), ("x", DROP), (".", NOOP), (" ", NOOP),
    ],
)
def test_key_mapping(key, action):
    assert play.key_to_action(key) == action


def test_quit_and_unknown_keys():
    assert play.key_to_action("q") == play.QUIT
    assert play.key_to_action(None) == play.QUIT  # EOF
    assert play.key_to_action("\x03") == play.QUIT
    assert play.key_to_action("z") is None


# --------------------------------------------------------------- I1 session
def test_play_session_with_scripted_keys():
    game = Game(load_fixed_layout("T0"), 1_000_000)  # spawns on row 1, above the rack row
    keys = iter(["s", "p", "z", "q"])
    out: list[str] = []
    traj = play.play(game, lambda: next(keys), out.append)
    text = "".join(out)
    assert len(traj.steps) == 2  # 'z' is ignored, 'q' quits
    assert traj.steps[1].picked >= 0
    assert "unknown key" in text and "legend:" in text and "order " in text and "carried " in text
    assert "score " in text and "quit" in text and "loop index" in text


def test_play_main_with_piped_input_records_a_replayable_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO("s\np\na\nd\nq\n"))
    path = tmp_path / "human.jsonl"
    assert play.main(["--layout", "T0", "--seed", "1000003", "--record", str(path)]) == 0
    out = capsys.readouterr().out
    assert "recorded to" in out
    traj = Trajectory.read(path)
    assert traj.header["player"] == "human" and len(traj.steps) == 4
    verify_replay(traj)


def test_play_runs_to_the_end_of_the_episode():
    game = Game(load_fixed_layout("T0"), 1_000_000)
    out: list[str] = []
    traj = play.play(game, lambda: ".", out.append)
    assert len(traj.steps) == 200 and "episode over" in "".join(out)


def test_line_key_reader_skips_blank_lines_and_handles_eof():
    r = play.LineKeyReader(io.StringIO("\n\nwasd\n\x1b[A\n"))
    assert r() == "w" and r() == "\x1b[A" and r() is None


# ------------------------------------------------------------------- I3
def test_demo_prints_a_movie_and_metrics():
    out = io.StringIO()
    assert demo.main(["--player", "oracle", "--layout", "T0", "--seed", "1000000"], out=out) == 0
    text = out.getvalue()
    assert text.count("step ") == 201  # initial frame + 200 steps
    assert "orders completed" in text and "loop index" in text


def test_demo_greedy_on_t0_gets_stuck(tmp_path):
    out = io.StringIO()
    path = tmp_path / "greedy.jsonl"
    demo.main(["--player", "greedy_straight", "--layout", "T0", "--final-only", "--record", str(path)], out=out)
    text = out.getvalue()
    assert "score 0 " in text and text.count("step ") == 1
    verify_replay(Trajectory.read(path))


def test_demo_rejects_bad_arguments():
    with pytest.raises(SystemExit):
        demo.main(["--every", "0"], out=io.StringIO())
    with pytest.raises(SystemExit):
        demo.main(["--player", "nobody"], out=io.StringIO())


def test_demo_runs_as_a_module():
    r = subprocess.run(
        [sys.executable, "-m", "weightshift.cli.demo", "--layout", "probe", "--seed", "7", "--final-only"],
        capture_output=True, text=True, timeout=60, cwd=ROOT,
    )
    assert r.returncode == 0, r.stderr
    assert "player oracle" in r.stdout and "orders completed" in r.stdout


# ------------------------------------------------------------------- H2
def test_bench_script_runs():
    spec = importlib.util.spec_from_file_location("bench_steps", ROOT / "scripts" / "bench_steps.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    rate, steps = bench.bench("T0", seconds=0.2)
    assert steps > 0 and rate > 0
