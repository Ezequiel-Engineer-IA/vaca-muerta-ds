"""Etapa 3: modelo de producción del primer año y curva de declinación (Arps).

1. Regresión de la acumulada a 12 meses (boe) en función de la profundidad y las
   etapas de fractura (modelo base, lo pedido) y un modelo ampliado con rama
   horizontal, año de inicio y tipo de pozo para medir cuánto mejora.
2. Curva tipo: perfil mensual normalizado (fracción de la acumulada a 12 meses)
   ajustado con una hiperbólica de Arps. Multiplicada por la acumulada predicha
   da la curva de declinación esperada de un pozo nuevo.
"""
from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

from . import config
from .visualizacion import TINTA, TINTA_SEC, _estilo

OBJETIVO = "boe12"
FEATURES_BASE = ["profundidad", "etapas_fractura"]
FEATURES_AMPLIADO_NUM = FEATURES_BASE + ["longitud_rama_m", "anio_inicio"]
FEATURES_AMPLIADO_CAT = ["tipopozo"]


# ---------------------------------------------------------------- dataset
def dataset_modelo(pozos: pd.DataFrame) -> pd.DataFrame:
    falta = {"etapas_fractura", "profundidad"} - set(pozos.columns)
    if falta:
        raise KeyError(f"Faltan columnas para el modelo: {falta}. ¿Está data/raw/fractura.csv?")
    df = pozos.dropna(subset=[OBJETIVO, *FEATURES_BASE])
    df = df[df[OBJETIVO] > 0]
    # Recorte de colas extremas del objetivo (errores de carga / pozos de prueba).
    lo, hi = df[OBJETIVO].quantile([0.01, 0.99])
    return df[df[OBJETIVO].between(lo, hi)].sort_values("fecha_inicio").reset_index(drop=True)


def _split_temporal(df: pd.DataFrame, frac_test: float = 0.2):
    """Los pozos más nuevos quedan para test: simula predecir pozos futuros."""
    corte = int(len(df) * (1 - frac_test))
    return df.iloc[:corte], df.iloc[corte:]


# ---------------------------------------------------------------- modelos
def _log_target(regresor) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(regressor=regresor, func=np.log1p, inverse_func=np.expm1)


def construir_modelos() -> dict[str, tuple[list[str], object]]:
    lineal_base = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("esc", StandardScaler()),
        ("reg", RidgeCV(alphas=np.logspace(-3, 3, 13))),
    ])
    gbr_base = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=15,
                                             min_samples_leaf=20, l2_regularization=1.0,
                                             random_state=42)
    prep_ampliado = ColumnTransformer([
        ("num", "passthrough", FEATURES_AMPLIADO_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), FEATURES_AMPLIADO_CAT),
    ])
    gbr_ampliado = Pipeline([
        ("prep", prep_ampliado),
        ("reg", HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_leaf_nodes=20,
                                              min_samples_leaf=20, l2_regularization=1.0,
                                              random_state=42)),
    ])
    feats_ampliado = FEATURES_AMPLIADO_NUM + FEATURES_AMPLIADO_CAT
    return {
        "Ridge polinómico (profundidad + etapas)": (FEATURES_BASE, _log_target(lineal_base)),
        "Gradient Boosting (profundidad + etapas)": (FEATURES_BASE, _log_target(gbr_base)),
        "Gradient Boosting ampliado (+ rama, año, tipo)": (feats_ampliado, _log_target(gbr_ampliado)),
    }


def _metricas(y, pred) -> dict[str, float]:
    return {"R2": r2_score(y, pred), "MAE_kboe": mean_absolute_error(y, pred) / 1_000,
            "MAPE_%": 100 * mean_absolute_percentage_error(y, pred)}


def entrenar(pozos: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = dataset_modelo(pozos)
    if "longitud_rama_m" not in df:
        df["longitud_rama_m"] = np.nan
    df["tipopozo"] = df["tipopozo"].fillna("SIN DATO").astype(str)
    train, test = _split_temporal(df)
    print(f"  {len(df):,} pozos con 12 meses + profundidad + etapas  "
          f"(train {len(train):,} hasta {train['fecha_inicio'].max():%Y-%m}, test {len(test):,})")

    filas, ajustados = [], {}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for nombre, (feats, modelo) in construir_modelos().items():
        cv = cross_val_score(modelo, train[feats], train[OBJETIVO], cv=kf, scoring="r2")
        modelo.fit(train[feats], train[OBJETIVO])
        pred = modelo.predict(test[feats])
        filas.append({"modelo": nombre, "R2_cv_train": cv.mean(), **_metricas(test[OBJETIVO], pred)})
        ajustados[nombre] = (feats, modelo, pred)

    resultados = pd.DataFrame(filas).set_index("modelo").round(3)
    config.DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    resultados.to_csv(config.DIR_MODELOS / "metricas.csv")

    # El modelo de producción pedido (profundidad + etapas): se elige el mejor de los dos
    # y se reentrena con todos los pozos.
    base = [n for n in ajustados if "ampliado" not in n]
    mejor = max(base, key=lambda n: resultados.loc[n, "R2"])
    feats, modelo, _ = ajustados[mejor]
    modelo.fit(df[feats], df[OBJETIVO])
    joblib.dump({"modelo": modelo, "features": feats, "nombre": mejor},
                config.DIR_MODELOS / "modelo_boe12.joblib")
    print(f"  Modelo guardado: {mejor}")

    _grafico_pred_vs_real(test, ajustados)
    _grafico_respuesta(df, modelo, feats)
    return resultados, ajustados


# ---------------------------------------------------------------- curva tipo (Arps)
def arps_hiperbolica(t, qi, di, b):
    return qi / np.power(1.0 + b * di * t, 1.0 / b)


def ajustar_curva_tipo(curvas: pd.DataFrame, meses: int = 36) -> dict:
    """P50 de la fracción mensual de la acumulada a 12 meses y ajuste Arps."""
    curvas = curvas.assign(frac=curvas["boe"] / curvas["boe12"])
    # Sólo pozos con la historia completa, para no sesgar la cola con pozos jóvenes.
    n_meses = curvas.groupby("idpozo")["mes_k"].max()
    completos = n_meses[n_meses >= meses - 1].index
    base = curvas[curvas["idpozo"].isin(completos)] if len(completos) >= 30 else curvas
    p50 = base.groupby("mes_k")["frac"].median().reindex(range(meses))
    p50 = p50.dropna()
    t = p50.index.to_numpy(dtype=float) + 0.5
    (qi, di, b), _ = curve_fit(arps_hiperbolica, t, p50.to_numpy(), p0=[0.15, 0.3, 1.0],
                               bounds=([1e-4, 1e-4, 0.01], [1.0, 10.0, 2.0]), maxfev=20_000)
    params = {"qi": qi, "di_mensual": di, "b": b, "meses": int(meses),
              "pozos_usados": int(base["idpozo"].nunique()), "p50": p50.round(5).tolist()}
    with open(config.DIR_MODELOS / "curva_tipo.json", "w") as f:
        json.dump(params, f, indent=2)
    print(f"  Curva tipo Arps: b = {b:.2f}, Di = {di:.3f} 1/mes ({params['pozos_usados']:,} pozos)")
    _grafico_curva_tipo(base, p50, params)
    return params


def predecir_pozo(profundidad: float, etapas: float, meses: int = 36) -> pd.DataFrame:
    """Acumulada del primer año y curva de declinación mensual esperada."""
    paquete = joblib.load(config.DIR_MODELOS / "modelo_boe12.joblib")
    with open(config.DIR_MODELOS / "curva_tipo.json") as f:
        curva = json.load(f)
    x = pd.DataFrame([{"profundidad": profundidad, "etapas_fractura": etapas}])[paquete["features"]]
    boe12 = float(paquete["modelo"].predict(x)[0])
    t = np.arange(meses) + 0.5
    frac = arps_hiperbolica(t, curva["qi"], curva["di_mensual"], curva["b"])
    frac = frac / frac[:12].sum()  # la curva suma exactamente la acumulada predicha en 12 meses
    vol = boe12 * frac
    dias = 30.4375
    return pd.DataFrame({"mes": np.arange(1, meses + 1), "boe_mes": vol.round(0),
                         "tasa_boe_d": (vol / dias).round(1), "acumulada_boe": vol.cumsum().round(0)})


# ---------------------------------------------------------------- figuras
def _grafico_pred_vs_real(test, ajustados) -> None:
    _estilo()
    fig, ejes = plt.subplots(1, len(ajustados), figsize=(5.2 * len(ajustados), 5), sharey=True)
    y = test[OBJETIVO] / 1_000
    lim = [0, float(np.nanpercentile(y, 99.5)) * 1.1]
    for ax, (nombre, (_, _, pred)) in zip(np.atleast_1d(ejes), ajustados.items()):
        ax.scatter(y, pred / 1_000, s=18, alpha=0.6, color="#2a78d6", edgecolor="white", lw=0.5)
        ax.plot(lim, lim, color=TINTA_SEC, ls="--", lw=1.2)
        ax.set(xlim=lim, ylim=lim, xlabel="Real (kboe, 12 meses)")
        ax.set_title(f"{nombre}\nR² test = {r2_score(y, pred / 1000):.2f}", fontsize=10)
    np.atleast_1d(ejes)[0].set_ylabel("Predicho (kboe, 12 meses)")
    fig.suptitle("Predicción de la acumulada del primer año — pozos más recientes (test)",
                 x=0.01, ha="left", fontweight="bold", color=TINTA)
    fig.tight_layout()
    fig.savefig(config.DIR_FIGURAS / "03_modelo_pred_vs_real.png")
    plt.close(fig)


def _grafico_respuesta(df, modelo, feats) -> None:
    """Superficie de respuesta del modelo base: acumulada vs etapas, por profundidad."""
    _estilo()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    etapas = np.linspace(*df["etapas_fractura"].quantile([0.02, 0.98]), 60)
    profs = df["profundidad"].quantile([0.1, 0.5, 0.9]).to_numpy()
    for prof, color, et in zip(profs, ["#2a78d6", "#eb6834", "#1baf7a"], ["P10", "P50", "P90"]):
        x = pd.DataFrame({"profundidad": prof, "etapas_fractura": etapas})[feats]
        pred = modelo.predict(x) / 1_000
        ax.plot(etapas, pred, lw=2, color=color, label=f"Profundidad {et}: {prof:,.0f} m")
        ax.annotate(f"{prof:,.0f} m", (etapas[-1], pred[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color=TINTA)
    ax.set(xlabel="Etapas de fractura", ylabel="Acumulada 12 meses predicha (kboe)")
    ax.set_title("Respuesta del modelo: etapas de fractura y profundidad", fontsize=12)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(config.DIR_FIGURAS / "04_modelo_respuesta.png")
    plt.close(fig)


def _grafico_curva_tipo(base, p50, params) -> None:
    _estilo()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    p10 = base.groupby("mes_k")["frac"].quantile(0.1).reindex(p50.index)
    p90 = base.groupby("mes_k")["frac"].quantile(0.9).reindex(p50.index)
    meses = p50.index + 1
    ax.fill_between(meses, p10 * 100, p90 * 100, color="#2a78d6", alpha=0.15, lw=0,
                    label="Banda P10–P90 de los pozos")
    ax.plot(meses, p50 * 100, "o", ms=5, color="#2a78d6", label="P50 observado")
    t = np.linspace(0.5, params["meses"] - 0.5, 200)
    ax.plot(t + 0.5, arps_hiperbolica(t, params["qi"], params["di_mensual"], params["b"]) * 100,
            color="#eb6834", lw=2,
            label=f"Arps hiperbólica (b = {params['b']:.2f}, Di = {params['di_mensual']:.2f}/mes)")
    ax.set(xlabel="Mes desde el inicio de producción",
           ylabel="Producción del mes (% de la acumulada a 12 meses)")
    ax.set_title("Curva tipo de declinación — pozos shale de Vaca Muerta", fontsize=12)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(config.DIR_FIGURAS / "05_curva_tipo_arps.png")
    plt.close(fig)
