"""The memorisation smoke test's memory cap, on /proc data whose answer is known.

The cap exists to stop a runaway search from holding the GPU queue.  Its first
version capped on VmRSS and did the opposite: it killed a healthy run at
22.2 GB after 2.9 minutes, because `PatchStore` memory-maps a 195 GiB latent
store and a streaming pass leaves that much clean page-cache resident.  Those
pages are file-backed and reclaimed the instant anything else wants the memory;
they cannot exhaust the machine, and a cap that counts them measures the size
of the store rather than the cost of the run.

These tests fix that distinction, because getting it wrong is silent in both
directions: too strict and the check never runs, too loose and a real leak
takes the machine down with the queue on it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analysis"))

import memorisation_smoke as MS  # noqa: E402


def write_status(tmp_path, *, anon_kb, file_kb, shmem_kb=0) -> int:
    """A /proc/<pid>/status good enough for the reader, under a fake pid."""
    pid = 424242
    proc = tmp_path / str(pid)
    proc.mkdir(parents=True)
    (proc / "status").write_text(
        "Name:\tpython\n"
        f"VmRSS:\t{anon_kb + file_kb + shmem_kb} kB\n"
        f"RssAnon:\t{anon_kb} kB\n"
        f"RssFile:\t{file_kb} kB\n"
        f"RssShmem:\t{shmem_kb} kB\n"
    )
    return pid


@pytest.fixture
def fake_proc(tmp_path, monkeypatch):
    """Point the reader at a /proc we control."""
    real = MS.Path

    def _make(**kw):
        pid = write_status(tmp_path, **kw)
        monkeypatch.setattr(
            MS, "Path",
            lambda p=".": real(str(p).replace("/proc/", f"{tmp_path}/")))
        return pid
    return _make


class TestRssSplit:

    def test_mapped_store_pages_do_not_count_towards_the_cap(self, fake_proc):
        """The exact shape of the run that was killed: all file, no anon."""
        pid = fake_proc(anon_kb=0, file_kb=22 * 1024 * 1024)
        anon, mapped = MS._rss_gb(pid)
        assert anon == pytest.approx(0.0)
        assert mapped == pytest.approx(22.0, abs=0.01)

    def test_real_memory_is_what_the_cap_sees(self, fake_proc):
        pid = fake_proc(anon_kb=6 * 1024 * 1024, file_kb=40 * 1024 * 1024)
        anon, mapped = MS._rss_gb(pid)
        assert anon == pytest.approx(6.0, abs=0.01)
        assert mapped == pytest.approx(40.0, abs=0.01)

    def test_shared_memory_counts_as_real(self, fake_proc):
        """Not backed by a file, so not reclaimable, so not free."""
        pid = fake_proc(anon_kb=1024 * 1024, file_kb=0, shmem_kb=2 * 1024 * 1024)
        anon, _ = MS._rss_gb(pid)
        assert anon == pytest.approx(3.0, abs=0.01)

    def test_a_dead_process_reads_zero_rather_than_raising(self):
        """The child exits while the poll loop is between reads."""
        assert MS._rss_gb(2 ** 30) == (0.0, 0.0)


class TestSmokeSet:

    def test_it_is_one_single_chunk_volume_and_one_multi_chunk_volume(self):
        """Both neighbour buckets, which is the point of the pair.

        A 192-cubed volume is one chunk and fills only the `present` bucket.
        Smoke-testing that alone would leave the UNKNOWN path — the thing the
        rewrite added — untouched.
        """
        assert MS.SMOKE_ASSESSMENTS == ("sampler", "multichunk")
