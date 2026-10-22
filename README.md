# Vaca Muerta: producción de pozos shale (Datos Abiertos de Hidrocarburos)

Proyecto de data science sobre la **producción por pozo de la Secretaría de Energía**
(Capítulo IV) y los **datos de fractura (Adjunto IV)**, aislando Vaca Muerta
(cuenca Neuquina + reservorio Shale).

## Estructura

```
main.py                 CLI del pipeline (descargar / preparar / graficos / modelo / todo / predecir)
src/config.py           rutas, columnas, conversiones de unidades, empresas
src/descarga.py         descarga de los CSV vía la API CKAN de datos.energia.gob.ar
src/limpieza.py         1. limpieza con pandas, filtro Vaca Muerta, tasa diaria, tabla por pozo
src/visualizacion.py    2. gráficos de líneas por empresa y scatter rama vs acumulada 12 meses
src/modelo.py           3. modelo scikit-learn de producción del 1er año + curva de declinación Arps
tests/test_pipeline.py  prueba de punta a punta con datos sintéticos (esquema oficial)
```

## Cómo correrlo

```bash
pip install -r requirements.txt
python main.py descargar --desde 2011       # baja produccion_<anio>.csv y fractura.csv a data/raw/
python main.py todo                         # limpieza + gráficos + modelo
python main.py predecir --profundidad 3000 --etapas 45
pytest -q
```

Si el portal no responde, descargá a mano desde <https://datos.energia.gob.ar> los CSV
anuales de *"Producción de petróleo y gas por pozo (Capítulo IV)"* como
`data/raw/produccion_<anio>.csv` y el de *"Datos de fractura de pozos de hidrocarburos —
Adjunto IV"* como `data/raw/fractura.csv`.

## 1. Limpieza y preparación (`src/limpieza.py`)

- Los CSV anuales se leen **en bloques** y se filtra en cada bloque `cuenca == NEUQUINA`
  y `sub_tipo_recurso == SHALE` (los archivos completos pesan cientos de MB).
  `--solo-formacion-vm` agrega `formacion` contiene "VACA MUERTA".
- Tipos numéricos, textos en mayúsculas y sin tildes, meses válidos, producciones
  negativas → 0, profundidades fuera de 500–8000 m → NaN, un registro por pozo-mes
  (se conserva la última rectificación).
- **Tasa diaria**:
  - `tasa_*_cal` = volumen mensual / días calendario. Es la que se suma entre pozos.
  - `tasa_*_tef` = volumen mensual / días efectivos (TEF, recortado al mes). Es la capacidad del pozo.
  - Unidades: petróleo m³/d (y bbl/d), gas Mm³/d, y **boe/d** = petróleo m³×6,29 + gas Mm³×5,89.
- Tabla por pozo: operador, área, tipo de pozo, profundidad (mediana), fecha del primer
  mes productivo y **acumulada de los primeros 12 meses** (sólo pozos con 12 meses
  calendario observados). Se cruza por `idpozo` con el Adjunto IV
  (longitud de rama horizontal y cantidad de etapas de fractura).

Salidas: `data/processed/vm_mensual.csv`, `vm_pozos.csv`, `serie_por_empresa.csv`.

## 2. Visualización (`reports/figures/`)

| Archivo | Contenido |
|---|---|
| `01_produccion_por_empresa.png` | Líneas: petróleo (kbbl/d) y gas (MMm³/d) de YPF, Vista, Shell y Pan American, últimos 10 años |
| `02_rama_vs_acumulada_12m.png` | Scatter: longitud de rama horizontal vs acumulada 12 meses (kboe), por tipo de pozo, con tendencia y ρ de Spearman |

Las empresas se agrupan por expresión regular sobre la razón social (`config.EMPRESAS`):
Shell opera como *O&G Developments Ltd S.A.* y Vista figuró como *Vista Oil & Gas* y
*Vista Energy*. Revisá `vm_mensual.csv['empresa'].unique()` si aparece una razón social nueva.

## 3. Modelo (`src/modelo.py`)

**Objetivo:** acumulada del primer año (boe) a partir de **profundidad y etapas de fractura**.

- Objetivo en escala log (`TransformedTargetRegressor`), recorte de percentiles 1–99.
- **Validación temporal**: el 20 % de pozos más recientes es el test (simula predecir pozos
  futuros) + validación cruzada 5-fold en train.
- Modelos comparados:
  1. Ridge con términos polinómicos de grado 2 (profundidad + etapas)
  2. HistGradientBoosting (profundidad + etapas)
  3. HistGradientBoosting ampliado (+ rama horizontal, año de inicio, tipo de pozo), como
     referencia de cuánto explican sólo las dos variables pedidas.
- Se guarda el mejor de 1 y 2, reentrenado con todos los pozos: `models/modelo_boe12.joblib`.
  Métricas en `models/metricas.csv`; figuras `03_modelo_pred_vs_real.png`, `04_modelo_respuesta.png`.

**Curva de declinación:** la mediana (P50) de la producción mensual normalizada por la
acumulada a 12 meses se ajusta con una **hiperbólica de Arps** `q(t) = qi / (1 + b·Di·t)^(1/b)`
(`models/curva_tipo.json`, figura `05_curva_tipo_arps.png`). `python main.py predecir`
combina ambos: acumulada predicha × curva tipo = perfil mensual esperado a 36 meses.

## Supuestos y limitaciones

- `profundidad` es la informada en el Capítulo IV; en pozos horizontales puede ser
  profundidad medida y no vertical: conviene revisarla antes de interpretar el modelo.
- Mezclar pozos petrolíferos y gasíferos en boe agrega varianza que profundidad y etapas
  no explican; el modelo ampliado incluye el tipo de pozo. Una alternativa es entrenar
  un modelo por fluido.
- Etapas y rama están muy correlacionadas (más rama → más etapas).
- Los nombres de columnas del Adjunto IV se resuelven con alias (`config.ALIAS_FRACTURA`);
  si el portal los cambia, el error lista las columnas disponibles.
