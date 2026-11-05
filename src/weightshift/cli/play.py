"""Play WeightShift by hand (acceptance I1).

    uv run python -m weightshift.cli.play --layout T0 --seed 1000000

Keys: arrows or w/a/s/d move, p pick, x drop (X marks the dock), . or space
no-op, q quit. When stdin is not a terminal, one key per line is read instead.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from ..env.constants import DOWN, DROP, LEFT, NOOP, PICK, RIGHT, UP
from ..env.game import Game
from ..env.render import render_frame
from ..metrics import compute_metrics
from ..trajectory import Recorder, Trajectory
from .common import LAYOUT_CHOICES, format_metrics, resolve_layout

KEYMAP = {
    "w": UP,
    "a": LEFT,
    "s": DOWN,
    "d": RIGHT,
    "\x1b[A": UP,
    "\x1b[B": DOWN,
    "\x1b[C": RIGHT,
    "\x1b[D": LEFT,
    "p": PICK,
    "x": DROP,
    ".": NOOP,
    " ": NOOP,
}
QUIT_KEYS = {"q", "\x03", "\x04"}  # q, Ctrl-C, Ctrl-D
QUIT = "quit"
HELP = "keys: arrows or w/a/s/d move   p pick   x drop   . no-op   q quit"


def key_to_action(key: str | None):
    """An action id, QUIT, or None for an unknown key. EOF (None) quits."""
    if key is None:
        return QUIT
    k = key if key.startswith("\x1b") else key.lower()
    if k in QUIT_KEYS:
        return QUIT
    return KEYMAP.get(k)


def play(
    game: Game,
    read_key: Callable[[], str | None],
    write: Callable[[str], None],
    *,
    clear_screen: bool = False,
) -> Trajectory:
    rec = Recorder(game, "human")
    last_action = last_reward = None
    note = ""
    while not game.done:
        frame = render_frame(game, last_action, last_reward)
        write(("\x1b[2J\x1b[H" if clear_screen else "") + frame + "\n" + HELP + (f"\n{note}" if note else "") + "\n")
        note = ""
        action = key_to_action(read_key())
        if action == QUIT:
            break
        if action is None:
            note = "unknown key"
            continue
        result = game.step(action)
        rec.record(action, result)
        last_action, last_reward = action, result.reward
    traj = rec.finish()
    write(render_frame(game, last_action, last_reward) + "\n")
    write(("episode over" if game.done else "quit") + "\n" + format_metrics(compute_metrics(traj)) + "\n")
    return traj


class TerminalKeyReader:
    """Reads single key presses (with arrow-key escape sequences) from a POSIX terminal."""

    def __init__(self, stream=None):
        self.stream = stream or sys.stdin

    def __enter__(self):
        import termios
        import tty

        self._fd = self.stream.fileno()
        self._old = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, *exc):
        import termios

        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def __call__(self) -> str | None:
        ch = self.stream.read(1)
        if not ch:
            return None
        if ch == "\x1b":
            ch += self.stream.read(2)
        return ch


class WindowsKeyReader:
    """Reads single key presses from a Windows console via msvcrt.

    Arrow keys arrive as a prefix ('\\xe0' or '\\x00') plus a scan code; they are
    translated to the same escape sequences the POSIX reader produces.
    """

    _ARROWS = {"H": "\x1b[A", "P": "\x1b[B", "M": "\x1b[C", "K": "\x1b[D"}

    def __init__(self, msvcrt_module=None):
        if msvcrt_module is None:
            import msvcrt as msvcrt_module  # Windows only
        self._msvcrt = msvcrt_module

    def __enter__(self):
        _enable_windows_ansi()
        return self

    def __exit__(self, *exc):
        return None

    def __call__(self) -> str | None:
        ch = self._msvcrt.getwch()
        if ch in ("\xe0", "\x00"):
            code = self._msvcrt.getwch()
            return self._ARROWS.get(code, "")  # other special keys (F1, Home, ...) are unknown keys
        if ch == "\x1a":  # Ctrl-Z, the Windows end-of-input key
            return None
        return ch


def _enable_windows_ansi() -> None:
    """Best effort: let legacy Windows consoles interpret the ANSI clear-screen sequence."""
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    except Exception:  # noqa: BLE001 - purely cosmetic
        pass


class LineKeyReader:
    """Reads one key per line (used when stdin is not a terminal, e.g. piped input or tests)."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


    def __init__(self, stream=None):
        self.stream = stream or sys.stdin

    def __call__(self) -> str | None:
        while True:
            line = self.stream.readline()
            if not line:
                return None
            key = line.rstrip("\n")
            if key == "":
                continue
            return key if key.startswith("\x1b") else key[0]


def make_key_reader(stream=None):
    """(reader, interactive): a single-key reader for a real terminal, else one key per line.

    POSIX terminals use termios, Windows consoles use msvcrt, and anything else
    (pipes, files, platforms without either module) falls back to line mode.
    """
    stream = stream or sys.stdin
    if not stream.isatty():
        return LineKeyReader(stream), False
    try:
        import termios  # noqa: F401
        import tty  # noqa: F401

        return TerminalKeyReader(stream), True
    except ImportError:
        pass
    try:
        return WindowsKeyReader(), True
    except ImportError:
        return LineKeyReader(stream), False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Play WeightShift in the terminal.")
    ap.add_argument("--layout", default="T0", choices=LAYOUT_CHOICES)
    ap.add_argument("--seed", type=int, default=1_000_000, help="episode seed (also the layout seed for families)")
    ap.add_argument("--role", choices=["train", "diagnosis", "eval", "select"], default=None)
    ap.add_argument("--record", default=None, help="write the episode to this JSONL file")
    args = ap.parse_args(argv)

    layout = resolve_layout(args.layout, args.seed, args.role)
    game = Game(layout, args.seed, role=args.role)

    def write(text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()

    reader, interactive = make_key_reader()
    with reader:
        traj = play(game, reader, write, clear_screen=interactive)
    if args.record:
        path = traj.write(args.record)
        write(f"recorded to {path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
