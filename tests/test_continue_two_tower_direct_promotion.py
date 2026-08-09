from scripts.continue_two_tower_direct_promotion import read_probe


def test_read_probe_returns_none_for_missing_file(tmp_path) -> None:
    assert read_probe(tmp_path / "missing.json") is None


def test_read_probe_reads_completed_status(tmp_path) -> None:
    path = tmp_path / "probe.json"
    path.write_text('{"status": "completed", "run_id": "probe"}')
    assert read_probe(path) == {"status": "completed", "run_id": "probe"}
