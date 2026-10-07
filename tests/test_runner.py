import csv

from acfc.core.runner import RunConfig, Runner, list_images, make_batches


def _collect(cfg):
    runner = Runner(cfg)
    return runner, list(runner.events())


def test_make_batches_de_5():
    assert make_batches(list(range(12)), 5) == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11]]
    assert make_batches([], 5) == []


def test_list_images_filtra_extensiones(tmp_path, make_image):
    make_image(tmp_path / "b.jpg"); make_image(tmp_path / "a.PNG"); (tmp_path / "x.txt").write_text("x")
    assert [p.name for p in list_images(tmp_path)] == ["a.PNG", "b.jpg"]


def test_corrida_completa_con_backend_fake(tmp_path, make_image):
    for i in range(7):
        make_image(tmp_path / f"ok{i}.jpg")
    make_image(tmp_path / "blanca.jpg", color=(255, 0, 0))
    (tmp_path / "rota.jpg").write_bytes(b"basura")
    cfg = RunConfig(tmp_path, workers=2, backend="fake")
    _, events = _collect(cfg)

    assert events[0]["type"] == "start" and events[0]["total"] == 9
    assert events[-1]["type"] == "finished"
    counts = events[-1]["counts"]
    assert counts == {"saved": 7, "discarded": 1, "error": 1}
    assert {e["worker"] for e in events if e["type"] == "worker_done"} == {0, 1}
    assert max(e["size"] for e in events if e["type"] == "batch_start") <= 5

    out = tmp_path / "outputs"
    assert sorted(p.name for p in out.glob("*.jpg")) == sorted(f"ok{i}.jpg" for i in range(7))
    desc = list(csv.DictReader((out / "log_descartadas.csv").open(encoding="utf-8")))
    assert [r["archivo"] for r in desc] == ["blanca.jpg"]
    err = list(csv.DictReader((out / "log_errores.csv").open(encoding="utf-8")))
    assert [r["archivo"] for r in err] == ["rota.jpg"]


def test_force_rehace_salidas_y_reinicia_los_logs(tmp_path, make_image):
    make_image(tmp_path / "ok.jpg")
    make_image(tmp_path / "blanca.jpg", color=(255, 0, 0))
    _collect(RunConfig(tmp_path, workers=1, backend="fake"))
    _, events = _collect(RunConfig(tmp_path, workers=1, backend="fake", force=True))
    assert events[-1]["counts"] == {"saved": 1, "discarded": 1}          # rehizo ok.jpg, no lo omitio
    desc = list(csv.DictReader((tmp_path / "outputs" / "log_descartadas.csv").open(encoding="utf-8")))
    assert [r["archivo"] for r in desc] == ["blanca.jpg"]                # sin filas duplicadas de la corrida anterior


def test_carpeta_vacia_termina_limpio(tmp_path):
    _, events = _collect(RunConfig(tmp_path, workers=3, backend="fake"))
    assert [e["type"] for e in events] == ["start", "finished"]
    assert events[-1]["counts"] == {}


def test_menos_lotes_que_workers(tmp_path, make_image):
    make_image(tmp_path / "a.jpg")
    _, events = _collect(RunConfig(tmp_path, workers=4, backend="fake"))
    assert events[0]["workers"] == 1
    assert events[-1]["counts"] == {"saved": 1}


def test_cancelar_antes_de_empezar_no_deja_tmp(tmp_path, make_image):
    for i in range(10):
        make_image(tmp_path / f"i{i}.jpg")
    runner = Runner(RunConfig(tmp_path, workers=2, backend="fake"))
    runner.cancel()
    events = list(runner.events())
    assert events[-1]["type"] == "finished" and events[-1]["cancelled"] is True
    assert not list((tmp_path / "outputs").glob(".*.tmp"))
