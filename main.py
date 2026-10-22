"""Pipeline Vaca Muerta: descarga -> limpieza -> visualización -> modelo.

Uso:
    python main.py descargar --desde 2011          # baja los CSV oficiales a data/raw/
    python main.py todo                            # limpieza + gráficos + modelo
    python main.py predecir --profundidad 3000 --etapas 45
"""
import argparse
import datetime as dt

import pandas as pd

from src import config, descarga, limpieza, modelo, visualizacion


def cargar_procesados():
    mensual = pd.read_csv(config.DIR_PROCESADO / "vm_mensual.csv", parse_dates=["fecha"])
    pozos = pd.read_csv(config.DIR_PROCESADO / "vm_pozos.csv", parse_dates=["fecha_inicio"])
    return mensual, pozos


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("descargar", help="Descarga los CSV del portal de datos abiertos")
    d.add_argument("--desde", type=int, default=2011)
    d.add_argument("--hasta", type=int, default=dt.date.today().year)
    d.add_argument("--forzar", action="store_true")

    for nombre in ("preparar", "todo"):
        s = sub.add_parser(nombre)
        s.add_argument("--solo-formacion-vm", action="store_true",
                       help="Además de Shale, exigir formación = Vaca Muerta")
    sub.add_parser("graficos")
    sub.add_parser("modelo")

    pr = sub.add_parser("predecir", help="Acumulada 1er año y curva de declinación de un pozo")
    pr.add_argument("--profundidad", type=float, required=True, help="metros")
    pr.add_argument("--etapas", type=float, required=True, help="etapas de fractura")
    pr.add_argument("--meses", type=int, default=36)

    a = p.parse_args()
    if a.cmd == "descargar":
        print("Descargando producción por pozo...")
        descarga.descargar_produccion(list(range(a.desde, a.hasta + 1)), a.forzar)
        print("Descargando datos de fractura...")
        descarga.descargar_fractura(a.forzar)
        return
    if a.cmd == "predecir":
        curva = modelo.predecir_pozo(a.profundidad, a.etapas, a.meses)
        print(f"Acumulada primer año estimada: {curva['acumulada_boe'].iat[11]:,.0f} boe")
        print(curva.to_string(index=False))
        return

    config.DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    if a.cmd in ("preparar", "todo"):
        mensual, pozos = limpieza.preparar(a.solo_formacion_vm)
    else:
        mensual, pozos = cargar_procesados()

    if a.cmd in ("graficos", "todo"):
        print("Generando gráficos...")
        visualizacion.generar_reporte(mensual, pozos)
    if a.cmd in ("modelo", "todo"):
        print("Entrenando modelos...")
        resultados, _ = modelo.entrenar(pozos)
        print(resultados.to_string())
        modelo.ajustar_curva_tipo(limpieza.curvas_normalizadas(mensual, pozos))


if __name__ == "__main__":
    main()
