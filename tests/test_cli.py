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
