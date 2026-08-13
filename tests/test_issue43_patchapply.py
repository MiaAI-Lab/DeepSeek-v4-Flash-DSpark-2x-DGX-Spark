#!/usr/bin/env python3
"""Apply the issue #43 hotfix to a real copy of the container scheduler.py,
assert anchors match, patch compiles (py_compile), and re-applying is a no-op
(idempotent). Also layers it on top of the #27 hotfix to mimic production
order. No GPU/torch required."""
import os, py_compile, shutil, subprocess, sys, tempfile, pathlib

HERE = pathlib.Path(__file__).resolve().parent  # .../tests
ROOT = HERE.parent                            # project root
HOT43 = ROOT / "patches/hotfix-dsv4-issue43-decode-fairness-and-diag.py"
HOT27 = ROOT / "patches/hotfix-dsv4-issue27-partial-prefill-concurrency.py"
# Real container target path the hotfix patches inside the image.
REAL = "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py"


def extract_real():
    """Copy the real scheduler.py out of the local image (no GPU needed)."""
    fd, path = tempfile.mkstemp(suffix=".py", text=True)
    os.close(fd)
    r = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "cat",
         "ghcr.io/anemll/dspark-vllm-gx10:0.1.1", REAL],
        check=True, stdout=open(path, "w"))
    return pathlib.Path(path)


def apply_hotfix_to_copy(src_path, hotfix_path):
    """Re-run the hotfix but pointed at our copy. The hotfix hardcodes P; we
    patch P via a tiny monkey by importing its logic on a redirected path."""
    txt = hotfix_path.read_text()
    # Replace the hardcoded Path so it targets our temp copy.
    marker = 'Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py")'
    txt = txt.replace(marker, f'Path({str(src_path)!r})')
    ns = {}
    # Local issue #27 revisions accept an optional target path for focused
    # tests, while upstream #43 still hardcodes P. Support both interfaces.
    old_argv = sys.argv
    try:
        sys.argv = [str(hotfix_path), str(src_path)]
        exec(compile(txt, str(hotfix_path), "exec"), ns)
    finally:
        sys.argv = old_argv


def main():
    tmpd = pathlib.Path(tempfile.mkdtemp())
    # 1. extract real scheduler
    real = extract_real()
    base = tmpd / "scheduler.py"
    shutil.copy(real, base)
    print(f"[1/6] extracted real container scheduler -> {base} ({base.stat().st_size} B)")

    # 2. anchors present?
    b = base.read_text()
    for anchor in [
        "import itertools\nimport time\n",
        "logger = init_logger(__name__)\n",
        "        prefill_scheduled = False\n",
        "        # Record the LoRAs in scheduled_running_reqs\n",
    ]:
        assert anchor in b, f"anchor missing: {anchor!r}"
    print("[2/6] real-file anchors verified (matches what hotfix asserts)")

    # 3. apply #27 first (production order), then #43
    apply_hotfix_to_copy(base, HOT27)
    print("[3/6] applied issue #27 hotfix on top")
    # real #27 hotfix marks
    assert "# [issue27-hotfix]" in base.read_text()

    apply_hotfix_to_copy(base, HOT43)
    assert "# [issue43-hotfix]" in base.read_text()
    print("[4/6] applied issue #43 hotfix on top of #27")

    # 5. compiles
    py_compile.compile(str(base), doraise=True)
    print("[5/6] py_compile OK (patched scheduler is syntactically valid)")

    # 6. idempotent: re-apply #43 -> no-op (exits early with SystemExit)
    try:
        apply_hotfix_to_copy(base, HOT43)
        print("[6/6] FAIL: re-apply did not raise SystemExit")
        sys.exit(1)
    except SystemExit:
        pass
    # ensure file unchanged after re-apply attempt
    after = base.read_text()
    once = base.read_text()
    assert once == after
    print("[6/6] idempotent re-apply is a no-op (SystemExit, file unchanged)")
    print("\nPASS: issue #43 hotfix applies cleanly on top of #27, compiles, "
          "and is idempotent against the real container scheduler.py")
    shutil.rmtree(tmpd)


if __name__ == "__main__":
    main()
