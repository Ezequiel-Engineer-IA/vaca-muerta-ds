"""Descarga de los CSV oficiales desde el portal CKAN de datos.energia.gob.ar.

Si el portal no es accesible, descargá los archivos a mano y dejalos en data/raw/:
  - produccion_<anio>.csv  (uno por año, dataset "Producción de petróleo y gas por pozo")
  - fractura.csv           (dataset "Datos de fractura de pozos - Adjunto IV")
"""
from __future__ import annotations

import re

import requests

from . import config


def _ckan(accion: str, **params) -> dict:
    r = requests.get(f"{config.CKAN_API}/{accion}", params=params, timeout=60)
    r.raise_for_status()
    datos = r.json()
    if not datos.get("success"):
        raise RuntimeError(f"CKAN {accion} falló: {datos.get('error')}")
    return datos["result"]


def _paquete(id_dataset: str, busqueda: str | None = None) -> dict:
    try:
        return _ckan("package_show", id=id_dataset)
    except requests.HTTPError:
        if not busqueda:
            raise
        resultados = _ckan("package_search", q=busqueda, rows=5)["results"]
        if not resultados:
            raise RuntimeError(f"No se encontró el dataset '{busqueda}' en CKAN")
        return resultados[0]


def _bajar(url: str, destino) -> None:
    print(f"  -> {destino.name}  ({url})")
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(destino, "wb") as f:
            for bloque in r.iter_content(chunk_size=1 << 20):
                f.write(bloque)


def descargar_produccion(anios: list[int], forzar: bool = False) -> None:
    config.DIR_RAW.mkdir(parents=True, exist_ok=True)
    recursos = _paquete(config.DATASET_PRODUCCION)["resources"]
    for anio in anios:
        destino = config.DIR_RAW / f"produccion_{anio}.csv"
        if destino.exists() and not forzar:
            continue
        candidatos = [
            r for r in recursos
            if str(r.get("format", "")).upper() == "CSV"
            and re.search(rf"\b{anio}\b", f"{r.get('name', '')} {r.get('url', '')}")
        ]
        if not candidatos:
            print(f"  (sin recurso CSV para {anio})")
            continue
        _bajar(candidatos[0]["url"], destino)


def descargar_fractura(forzar: bool = False) -> None:
    config.DIR_RAW.mkdir(parents=True, exist_ok=True)
    destino = config.DIR_RAW / "fractura.csv"
    if destino.exists() and not forzar:
        return
    recursos = _paquete(config.DATASET_FRACTURA, config.BUSQUEDA_FRACTURA)["resources"]
    csvs = [r for r in recursos if str(r.get("format", "")).upper() == "CSV"]
    if not csvs:
        raise RuntimeError("El dataset de fractura no tiene recursos CSV")
    _bajar(csvs[0]["url"], destino)
