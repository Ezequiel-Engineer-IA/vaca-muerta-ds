"""Etapa 1: carga, limpieza, filtro de Vaca Muerta y normalización a tasa diaria."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config

NUMERICAS = ["anio", "mes", "profundidad", "prod_pet", "prod_gas", "prod_agua", "tef"]
TEXTO = ["empresa", "cuenca", "provincia", "areayacimiento", "formacion",
         "tipo_de_recurso", "sub_tipo_recurso", "tipopozo", "tipoextraccion"]


def _normalizar_texto(s: pd.Series) -> pd.Series:
    return (s.astype("string").str.strip().str.upper()
             .str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii"))


def _leer_filtrado(ruta: Path, chunksize: int) -> pd.DataFrame:
    """Lee un CSV del Capítulo IV en bloques y conserva sólo cuenca Neuquina + Shale."""
    pedidas = set(config.COLS_PRODUCCION)
    for encoding in ("utf-8", "latin-1"):
        try:
            partes = []
            lector = pd.read_csv(
                ruta, chunksize=chunksize, encoding=encoding, low_memory=False,
                usecols=lambda c: c.strip().lower() in pedidas,
            )
            for bloque in lector:
                bloque.columns = bloque.columns.str.strip().str.lower()
                cuenca = _normalizar_texto(bloque["cuenca"])
                subtipo = _normalizar_texto(bloque["sub_tipo_recurso"])
                partes.append(bloque[(cuenca == config.CUENCA) & (subtipo == config.SUBTIPO_RESERVORIO)])
            return pd.concat(partes, ignore_index=True) if partes else pd.DataFrame()
        except UnicodeDecodeError:
            continue
    raise ValueError(f"No se pudo decodificar {ruta}")


def cargar_produccion(dir_raw: Path | None = None, chunksize: int = 250_000) -> pd.DataFrame:
    dir_raw = dir_raw or config.DIR_RAW
    archivos = sorted(dir_raw.glob("produccion_*.csv"))
    if not archivos:
        raise FileNotFoundError(
            f"No hay archivos produccion_<anio>.csv en {dir_raw}. Corré `python main.py descargar`.")
    frames = []
    for ruta in archivos:
        df = _leer_filtrado(ruta, chunksize)
        print(f"  {ruta.name}: {len(df):,} registros Neuquina/Shale")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    faltan = set(config.COLS_PRODUCCION) - set(df.columns)
    if faltan:
        raise KeyError(f"Faltan columnas esperadas del Capítulo IV: {sorted(faltan)}")
    return df


def limpiar_produccion(df: pd.DataFrame, solo_formacion_vm: bool = False) -> pd.DataFrame:
    """Tipos, textos normalizados, duplicados, valores imposibles y fecha mensual."""
    df = df.copy()
    for c in NUMERICAS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in TEXTO:
        df[c] = _normalizar_texto(df[c])

    df = df.dropna(subset=["idpozo", "anio", "mes"])
    df = df[df["mes"].between(1, 12)]
    df["idpozo"] = df["idpozo"].astype("int64")
    df["fecha"] = pd.to_datetime(dict(year=df["anio"].astype(int), month=df["mes"].astype(int), day=1))

    # Producciones e inyecciones negativas son errores de carga.
    for c in ["prod_pet", "prod_gas", "prod_agua", "tef"]:
        df[c] = df[c].where(df[c] >= 0).fillna(0.0)
    df["profundidad"] = df["profundidad"].where(df["profundidad"].between(500, 8000))

    # Un registro por pozo-mes (las rectificaciones vuelven a informar el mismo mes).
    df = df.sort_values(["idpozo", "fecha"]).drop_duplicates(["idpozo", "fecha"], keep="last")

    if solo_formacion_vm:
        df = df[df["formacion"].str.contains("VACA MUERTA", na=False)]

    return df.reset_index(drop=True)


def calcular_tasas(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza la producción mensual a tasa diaria.

    - tasa_*_cal: volumen / días calendario del mes. Es la que se suma entre pozos
      (producción de una empresa o de la cuenca).
    - tasa_*_tef: volumen / días efectivos en producción (TEF). Es la capacidad real
      del pozo; si TEF falta o es 0 con producción > 0 se usa el mes calendario.
    """
    df = df.copy()
    dias_mes = df["fecha"].dt.days_in_month
    tef = df["tef"].clip(upper=dias_mes)
    tef = tef.where(tef > 0, dias_mes)

    df["boe"] = df["prod_pet"] * config.M3_A_BBL + df["prod_gas"] * config.MM3_GAS_A_BOE
    for nombre, col in [("pet_m3d", "prod_pet"), ("gas_mm3d", "prod_gas"), ("boe_d", "boe")]:
        df[f"tasa_{nombre}_cal"] = df[col] / dias_mes
        df[f"tasa_{nombre}_tef"] = df[col] / tef
    df["tasa_pet_bbld_cal"] = df["tasa_pet_m3d_cal"] * config.M3_A_BBL
    return df


def tabla_pozos(df: pd.DataFrame, meses: int = 12) -> pd.DataFrame:
    """Una fila por pozo: atributos y acumulada de los primeros `meses` meses.

    El mes 0 es el primer mes con producción (petróleo o gas). Sólo se calcula la
    acumulada para pozos con `meses` meses calendario observados desde su inicio.
    """
    prod = df[(df["prod_pet"] > 0) | (df["prod_gas"] > 0)]
    inicio = prod.groupby("idpozo")["fecha"].min().rename("fecha_inicio")
    df = df.join(inicio, on="idpozo").dropna(subset=["fecha_inicio"])
    df["mes_k"] = ((df["fecha"].dt.year - df["fecha_inicio"].dt.year) * 12
                   + df["fecha"].dt.month - df["fecha_inicio"].dt.month)

    primeros = df[df["mes_k"].between(0, meses - 1)]
    acum = primeros.groupby("idpozo").agg(
        np12_pet_m3=("prod_pet", "sum"),
        gp12_gas_mm3=("prod_gas", "sum"),
        boe12=("boe", "sum"),
    )
    attrs = df.sort_values("fecha").groupby("idpozo").agg(
        empresa=("empresa", "first"),
        area=("areayacimiento", "first"),
        formacion=("formacion", "first"),
        tipopozo=("tipopozo", lambda s: s.mode().iat[0] if s.notna().any() else pd.NA),
        profundidad=("profundidad", "median"),
        fecha_inicio=("fecha_inicio", "first"),
    )
    pozos = attrs.join(acum)
    pozos["np12_pet_bbl"] = pozos["np12_pet_m3"] * config.M3_A_BBL

    ultimo_mes = df["fecha"].max()
    completos = pozos["fecha_inicio"] + pd.DateOffset(months=meses - 1) <= ultimo_mes
    pozos.loc[~completos, ["np12_pet_m3", "gp12_gas_mm3", "boe12", "np12_pet_bbl"]] = np.nan
    pozos["anio_inicio"] = pozos["fecha_inicio"].dt.year
    return pozos.reset_index()


def curvas_normalizadas(df: pd.DataFrame, pozos: pd.DataFrame, meses: int = 36) -> pd.DataFrame:
    """Tasa diaria (boe/d calendario) por mes desde el inicio, para la curva tipo."""
    base = df.merge(pozos[["idpozo", "fecha_inicio", "boe12"]], on="idpozo")
    base["mes_k"] = ((base["fecha"].dt.year - base["fecha_inicio"].dt.year) * 12
                     + base["fecha"].dt.month - base["fecha_inicio"].dt.month)
    base = base[base["mes_k"].between(0, meses - 1) & base["boe12"].gt(0)]
    return base[["idpozo", "mes_k", "tasa_boe_d_cal", "boe", "boe12"]]


def _resolver_columnas(df: pd.DataFrame) -> dict[str, str]:
    cols = {c.strip().lower(): c for c in df.columns}
    mapeo = {}
    for destino, alias in config.ALIAS_FRACTURA.items():
        encontrada = next((cols[a] for a in alias if a in cols), None)
        if encontrada is not None:
            mapeo[encontrada] = destino
    for requerida in ("idpozo", "longitud_rama_m", "etapas_fractura"):
        if requerida not in mapeo.values():
            raise KeyError(f"El CSV de fractura no tiene la columna '{requerida}'. "
                           f"Columnas disponibles: {list(df.columns)}")
    return mapeo


def cargar_fractura(ruta: Path | None = None) -> pd.DataFrame:
    """Lee el Adjunto IV y deja una fila por pozo (rama y etapas máximas informadas)."""
    ruta = ruta or config.DIR_RAW / "fractura.csv"
    if not ruta.exists():
        raise FileNotFoundError(f"No existe {ruta}. Corré `python main.py descargar`.")
    try:
        raw = pd.read_csv(ruta, low_memory=False)
    except UnicodeDecodeError:
        raw = pd.read_csv(ruta, low_memory=False, encoding="latin-1")
    df = raw.rename(columns=_resolver_columnas(raw))
    df = df[[c for c in config.ALIAS_FRACTURA if c in df.columns]]

    for c in df.columns.drop(["fecha_fin_fractura"], errors="ignore"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["idpozo"])
    df["idpozo"] = df["idpozo"].astype("int64")
    # Rangos físicamente razonables para pozos horizontales de Vaca Muerta.
    df["longitud_rama_m"] = df["longitud_rama_m"].where(df["longitud_rama_m"].between(100, 6000))
    df["etapas_fractura"] = df["etapas_fractura"].where(df["etapas_fractura"].between(1, 150))

    agg = {"longitud_rama_m": "max", "etapas_fractura": "max"}
    agg.update({c: "sum" for c in ("arena_nacional_tn", "arena_importada_tn", "agua_inyectada_m3")
                if c in df.columns})
    return df.groupby("idpozo").agg(agg).reset_index()


def preparar(solo_formacion_vm: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pipeline completo de la etapa 1. Guarda los resultados en data/processed/."""
    config.DIR_PROCESADO.mkdir(parents=True, exist_ok=True)
    print("Cargando Capítulo IV (filtro: cuenca Neuquina + Shale)...")
    mensual = calcular_tasas(limpiar_produccion(cargar_produccion(), solo_formacion_vm))
    pozos = tabla_pozos(mensual)

    ruta_frac = config.DIR_RAW / "fractura.csv"
    if ruta_frac.exists():
        pozos = pozos.merge(cargar_fractura(ruta_frac), on="idpozo", how="left")
    else:
        print(f"  Aviso: falta {ruta_frac}; no habrá longitud de rama ni etapas de fractura.")

    mensual.to_csv(config.DIR_PROCESADO / "vm_mensual.csv", index=False)
    pozos.to_csv(config.DIR_PROCESADO / "vm_pozos.csv", index=False)
    print(f"  {mensual['idpozo'].nunique():,} pozos shale, {len(mensual):,} registros pozo-mes, "
          f"{mensual['fecha'].min():%Y-%m} a {mensual['fecha'].max():%Y-%m}")
    return mensual, pozos

