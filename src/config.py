"""Constantes del proyecto: rutas, nombres de columnas, conversiones y empresas."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DIR_RAW = RAIZ / "data" / "raw"
DIR_PROCESADO = RAIZ / "data" / "processed"
DIR_FIGURAS = RAIZ / "reports" / "figures"
DIR_MODELOS = RAIZ / "models"

# --- Datasets en el portal de Datos Abiertos de la Secretaría de Energía (CKAN) ---
CKAN_API = "https://datos.energia.gob.ar/api/3/action"
# Producción de petróleo y gas por pozo (Capítulo IV): un CSV por año.
DATASET_PRODUCCION = "produccion-de-petroleo-y-gas-por-pozo"
# Datos de fractura de pozos (Adjunto IV, Res. SE 2019): longitud de rama y etapas.
DATASET_FRACTURA = "datos-de-fractura-de-pozos-de-hidrocarburos-adjunto-iv-actualizacion-diaria"
BUSQUEDA_FRACTURA = "fractura adjunto IV"

# --- Columnas usadas del Capítulo IV ---
# Unidades oficiales: prod_pet [m3], prod_gas [Mm3 = miles de m3], tef [días].
COLS_PRODUCCION = [
    "idpozo", "anio", "mes", "empresa", "cuenca", "provincia",
    "areayacimiento", "formacion", "tipo_de_recurso", "sub_tipo_recurso",
    "tipopozo", "tipoextraccion", "profundidad",
    "prod_pet", "prod_gas", "prod_agua", "tef",
]

# --- Columnas del Adjunto IV (fractura). Se aceptan alias por si el portal las renombra.
ALIAS_FRACTURA = {
    "idpozo": ["idpozo", "id_pozo"],
    "longitud_rama_m": ["longitud_rama_horizontal_m", "longitud_rama_horizontal", "longitud_rama"],
    "etapas_fractura": ["cantidad_fracturas", "cantidad_etapas", "etapas_fractura"],
    "arena_nacional_tn": ["arena_bombeada_nacional_tn"],
    "arena_importada_tn": ["arena_bombeada_importada_tn"],
    "agua_inyectada_m3": ["agua_inyectada_m3"],
    "fecha_fin_fractura": ["fecha_fin_fractura", "fecha_fin"],
}

# --- Filtros que aíslan Vaca Muerta ---
CUENCA = "NEUQUINA"
SUBTIPO_RESERVORIO = "SHALE"

# --- Conversiones ---
M3_A_BBL = 6.28981
# 1 Mm3 de gas = 1000 m3 = 35.3147 Mcf ; 1 boe ~ 6 Mcf  ->  5.886 boe por Mm3
MM3_GAS_A_BOE = 35.3147 / 6.0

# --- Empresas a comparar: etiqueta -> regex sobre la razón social informada ---
# Shell opera en Argentina vía "O&G Developments Ltd S.A."; Vista figuró como
# "Vista Oil & Gas Argentina" y luego "Vista Energy Argentina".
EMPRESAS = {
    "YPF": r"\bYPF\b",
    "Vista": r"\bVISTA\b",
    "Shell": r"SHELL|O\s*&\s*G\s+DEVELOPMENTS",
    "Pan American": r"PAN\s+AMERICAN",
}
