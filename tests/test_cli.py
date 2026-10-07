from acfc.cli import main


def test_cli_procesa_carpeta_con_backend_fake(tmp_path, make_image):
    for i in range(6):
        make_image(tmp_path / f"i{i}.jpg")
    make_image(tmp_path / "blanca.jpg", color=(255, 0, 0))
    code = main([str(tmp_path), "--workers", "2", "--backend", "fake"])
    assert code == 0
    assert len(list((tmp_path / "outputs").glob("*.jpg"))) == 6


def test_cli_ruta_inexistente_devuelve_error(tmp_path, capsys):
    assert main([str(tmp_path / "nada"), "--backend", "fake"]) == 2
    assert "no existe" in capsys.readouterr().err.lower()


def test_cli_por_fases_es_el_modo_por_defecto(tmp_path, make_image):
    for i in range(4):
        make_image(tmp_path / f"i{i}.jpg", size=(100, 100))
    make_image(tmp_path / "blanca.jpg", size=(100, 100), color=(255, 0, 0))
    assert main([str(tmp_path), "--workers", "2", "--backend", "fake"]) == 0
    assert len(list((tmp_path / "fase1_mapa").glob("*.jpg"))) == 5
    assert len(list((tmp_path / "fase2_personas").glob("*.jpg"))) == 4
    assert len(list((tmp_path / "outputs").glob("*.jpg"))) == 4


def test_cli_una_sola_fase(tmp_path, make_image):
    make_image(tmp_path / "a.jpg", size=(100, 100))
    assert main([str(tmp_path), "--fase", "1", "--backend", "fake"]) == 0
    assert (tmp_path / "fase1_mapa" / "a.jpg").exists() and not (tmp_path / "outputs").exists()


def test_cli_modo_unico_conserva_el_comportamiento_anterior(tmp_path, make_image):
    for i in range(3):
        make_image(tmp_path / f"i{i}.jpg")
    assert main([str(tmp_path), "--modo", "unico", "--backend", "fake"]) == 0
    assert len(list((tmp_path / "outputs").glob("*.jpg"))) == 3 and not (tmp_path / "fase1_mapa").exists()


def test_cli_fase_sin_la_carpeta_anterior_da_error_claro(tmp_path, capsys):
    assert main([str(tmp_path), "--fase", "2", "--backend", "fake"]) == 2
    assert "falta la carpeta de la fase anterior" in capsys.readouterr().err
