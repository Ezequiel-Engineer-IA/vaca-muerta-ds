"""Prueba de humo del pipeline con datos SINTÉTICOS que imitan el esquema oficial.

Sólo verifica que el código corre de punta a punta; los números no representan
Vaca Muerta. Los resultados reales salen de `python main.py descargar && python main.py todo`.
"""
import numpy as np
import pandas as pd
import pytest

from src import config, limpieza, modelo, visualizacion

EMPRESAS = ["YPF S.A.", "VISTA ENERGY ARGENTINA SAU", "O&G DEVELOPMENTS LTD S.A.",
            "PAN AMERICAN ENERGY SL", "TECPETROL S.A."]


def _sintetico(rng, n_pozos=300):
    filas, frac = [], []
    for i in range(n_pozos):
        inicio = pd.Timestamp("2014-01-01") + pd.DateOffset(months=int(rng.integers(0, 110)))
        prof = rng.uniform(2500, 3500)
        etapas = int(rng.integers(10, 80))
        rama = etapas * rng.uniform(50, 70)
        petrolifero = rng.random() < 0.7
        qi = (etapas * 8 + prof * 0.05) * rng.lognormal(0, 0.25)
        for k in range(int(rng.integers(6, 60))):
            f = inicio + pd.DateOffset(months=k)
            if f > pd.Timestamp("2024-12-01"):
                break
            q = qi / (1 + 1.1 * 0.3 * (k + 0.5)) ** (1 / 1.1)
            filas.append({
                "idpozo": 1000 + i, "anio": f.year, "mes": f.month, "empresa": EMPRESAS[i % 5],
                "cuenca": "NEUQUINA", "provincia": "NEUQUEN", "areayacimiento": "LOMA X",
                "formacion": "vaca muerta", "tipo_de_recurso": "NO CONVENCIONAL",
                "sub_tipo_recurso": "SHALE", "tipopozo": "Petrolífero" if petrolifero else "Gasífero",
                "tipoextraccion": "Surgencia Natural", "profundidad": prof,
                "prod_pet": q * 30 if petrolifero else q * 3, "prod_gas": q * 20 if petrolifero else q * 150,
                "prod_agua": q, "tef": 30,
            })
        frac.append({"idpozo": 1000 + i, "longitud_rama_horizontal_m": rama,
                     "cantidad_fracturas": etapas, "arena_bombeada_nacional_tn": etapas * 200})
    df = pd.DataFrame(filas)
    # Ruido: un registro convencional de otra cuenca, que el filtro debe descartar.
    ruido = df.iloc[[0]].assign(cuenca="GOLFO SAN JORGE", sub_tipo_recurso="", idpozo=1)
    return pd.concat([df, ruido]), pd.DataFrame(frac)


@pytest.fixture()
def entorno(tmp_path, monkeypatch):
    for attr, sub in [("DIR_RAW", "raw"), ("DIR_PROCESADO", "proc"),
                      ("DIR_FIGURAS", "fig"), ("DIR_MODELOS", "mod")]:
        (tmp_path / sub).mkdir()
        monkeypatch.setattr(config, attr, tmp_path / sub)
    prod, frac = _sintetico(np.random.default_rng(0))
    for anio, g in prod.groupby("anio"):
        g.to_csv(tmp_path / "raw" / f"produccion_{anio}.csv", index=False)
    frac.to_csv(tmp_path / "raw" / "fractura.csv", index=False)
    return tmp_path


def test_pipeline_completo(entorno):
    mensual, pozos = limpieza.preparar()
    assert set(mensual["cuenca"]) == {"NEUQUINA"}
    assert 1 not in set(mensual["idpozo"])
    assert (mensual["tasa_pet_m3d_cal"] >= 0).all()
    assert pozos["longitud_rama_m"].notna().any()

    visualizacion.generar_reporte(mensual, pozos)
    resultados, _ = modelo.entrenar(pozos)
    assert len(resultados) == 3
    modelo.ajustar_curva_tipo(limpieza.curvas_normalizadas(mensual, pozos))

    curva = modelo.predecir_pozo(3000, 40)
    assert curva["acumulada_boe"].iat[11] > 0
    assert sorted(p.name for p in config.DIR_FIGURAS.glob("*.png")) == [
        "01_produccion_por_empresa.png", "02_rama_vs_acumulada_12m.png",
        "03_modelo_pred_vs_real.png", "04_modelo_respuesta.png", "05_curva_tipo_arps.png"]


def test_tasa_diaria_usa_tef_y_calendario():
    df = pd.DataFrame({"fecha": pd.to_datetime(["2024-02-01"]), "prod_pet": [290.0],
                       "prod_gas": [0.0], "tef": [10.0]})
    t = limpieza.calcular_tasas(df)
    assert t["tasa_pet_m3d_cal"].iat[0] == pytest.approx(10.0)   # 290 / 29 días
    assert t["tasa_pet_m3d_tef"].iat[0] == pytest.approx(29.0)   # 290 / 10 días efectivos


def test_grupos_empresa():
    s = pd.Series(["YPF S.A.", "O&G DEVELOPMENTS LTD S.A.", "VISTA OIL & GAS ARGENTINA SAU",
                   "PAN AMERICAN ENERGY SL", "TOTAL AUSTRAL"])
    assert visualizacion.asignar_grupo_empresa(s).tolist()[:4] == ["YPF", "Shell", "Vista", "Pan American"]
