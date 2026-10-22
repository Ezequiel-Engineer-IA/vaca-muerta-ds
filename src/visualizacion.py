"""Etapa 2: gráficos de evolución por empresa y rama horizontal vs acumulada a 12 meses."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from . import config

# Paleta categórica en orden fijo (validada para daltonismo en pares adyacentes).
COLORES_EMPRESA = {
    "YPF": "#2a78d6",
    "Vista": "#eb6834",
    "Shell": "#1baf7a",
    "Pan American": "#eda100",
}
COLORES_FLUIDO = ["#2a78d6", "#eb6834", "#1baf7a"]
TINTA = "#0b0b0b"
TINTA_SEC = "#52514e"
GRILLA = "#e4e3df"


def _estilo() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "axes.edgecolor": GRILLA, "grid.color": GRILLA, "grid.linewidth": 0.8,
        "axes.labelcolor": TINTA_SEC, "xtick.color": TINTA_SEC, "ytick.color": TINTA_SEC,
        "axes.titlecolor": TINTA, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110,
        "savefig.dpi": 160, "savefig.bbox": "tight",
    })


def asignar_grupo_empresa(empresa: pd.Series) -> pd.Series:
    grupo = pd.Series(pd.NA, index=empresa.index, dtype="string")
    for etiqueta, patron in config.EMPRESAS.items():
        grupo = grupo.mask(grupo.isna() & empresa.str.contains(patron, regex=True, na=False), etiqueta)
    return grupo


def _rotular_finales(ax, finales, sep: float = 0.05) -> None:
    """Rótulo directo al final de cada línea, separando los que quedan superpuestos."""
    y0, y1 = ax.get_ylim()
    minimo = sep * (y1 - y0)
    finales = sorted(finales, key=lambda f: f[2])
    posiciones = []
    for _, _, y in finales:
        posiciones.append(max(y, posiciones[-1] + minimo) if posiciones else y)
    for (grupo, x, y), y_txt in zip(finales, posiciones):
        ax.annotate(f"{grupo}  {y:,.1f}", (x, y_txt), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=TINTA, annotation_clip=False)


def serie_por_empresa(mensual: pd.DataFrame, anios: int = 10) -> pd.DataFrame:
    """Tasa diaria (calendario) sumada por empresa y mes, últimos `anios` años."""
    df = mensual.assign(grupo=asignar_grupo_empresa(mensual["empresa"])).dropna(subset=["grupo"])
    desde = mensual["fecha"].max() - pd.DateOffset(years=anios)
    df = df[df["fecha"] > desde]
    serie = (df.groupby(["grupo", "fecha"])
               .agg(pet_kbbld=("tasa_pet_bbld_cal", "sum"), gas_mmm3d=("tasa_gas_mm3d_cal", "sum"))
               .reset_index())
    serie["pet_kbbld"] /= 1_000
    serie["gas_mmm3d"] /= 1_000  # Mm3/d -> millones de m3/d
    return serie


def grafico_produccion_empresas(mensual: pd.DataFrame, anios: int = 10):
    _estilo()
    serie = serie_por_empresa(mensual, anios)
    fig, ejes = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)
    paneles = [("pet_kbbld", "Petróleo — miles de bbl/d"), ("gas_mmm3d", "Gas — millones de m³/d")]
    for ax, (col, titulo) in zip(ejes, paneles):
        finales = []
        for grupo, color in COLORES_EMPRESA.items():
            s = serie[serie["grupo"] == grupo].sort_values("fecha")
            if s.empty:
                continue
            ax.plot(s["fecha"], s[col], color=color, lw=2, label=grupo)
            finales.append((grupo, s["fecha"].iat[-1], s[col].iat[-1]))
        ax.set_ylim(bottom=0)
        _rotular_finales(ax, finales)
        ax.set_title(titulo, fontsize=11)
        ax.set_ylabel("")
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ejes[0].legend(loc="upper left", frameon=False, ncol=4)
    ejes[-1].set_xlabel("")
    fig.suptitle("Vaca Muerta (Neuquina · Shale): producción por operador, últimos "
                 f"{anios} años", x=0.01, ha="left", fontsize=13, fontweight="bold", color=TINTA)
    fig.text(0.01, -0.01, "Tasa diaria calendario = volumen mensual / días del mes, sumada por "
             "operador. Fuente: Secretaría de Energía, Capítulo IV.", fontsize=8, color=TINTA_SEC)
    fig.tight_layout()
    return fig, serie


def grafico_rama_vs_acumulada(pozos: pd.DataFrame):
    _estilo()
    df = pozos.dropna(subset=["longitud_rama_m", "boe12"])
    df = df[df["boe12"] > 0].copy()
    df["kboe12"] = df["boe12"] / 1_000
    df["tipopozo"] = df["tipopozo"].fillna("SIN DATO").str.title()

    fig, ax = plt.subplots(figsize=(10, 6.5))
    tipos = df["tipopozo"].value_counts().index[:3]  # máx. 3 categorías en un scatter
    df.loc[~df["tipopozo"].isin(tipos), "tipopozo"] = "Otro"
    sns.scatterplot(data=df, x="longitud_rama_m", y="kboe12", hue="tipopozo",
                    hue_order=list(tipos), palette=COLORES_FLUIDO[:len(tipos)],
                    s=28, alpha=0.65, edgecolor="white", linewidth=0.6, ax=ax)

    # Tendencia lineal global y correlación de rangos.
    if len(df) > 2:
        m, b = np.polyfit(df["longitud_rama_m"], df["kboe12"], 1)
        xs = np.linspace(df["longitud_rama_m"].min(), df["longitud_rama_m"].max(), 50)
        ax.plot(xs, m * xs + b, color=TINTA_SEC, lw=1.5, ls="--",
                label=f"Tendencia: +{m * 1000:,.0f} kboe por km de rama")
        rho = df[["longitud_rama_m", "kboe12"]].corr(method="spearman").iat[0, 1]
        ax.text(0.99, 0.02, f"n = {len(df):,} pozos · Spearman ρ = {rho:.2f}",
                transform=ax.transAxes, ha="right", fontsize=9, color=TINTA_SEC)

    ax.set_title("Longitud de rama horizontal vs producción acumulada a 12 meses", fontsize=12)
    ax.set_xlabel("Longitud de rama horizontal (m)")
    ax.set_ylabel("Acumulada primeros 12 meses (miles de boe)")
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.legend(title="Tipo de pozo", frameon=False, loc="upper left")
    fig.text(0.01, -0.02, "boe = petróleo (m³ × 6,29) + gas (Mm³ × 5,89). Pozos con 12 meses "
             "completos desde su primer mes productivo.", fontsize=8, color=TINTA_SEC)
    fig.tight_layout()
    return fig


def generar_reporte(mensual: pd.DataFrame, pozos: pd.DataFrame) -> None:
    config.DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    fig, serie = grafico_produccion_empresas(mensual)
    fig.savefig(config.DIR_FIGURAS / "01_produccion_por_empresa.png")
    serie.to_csv(config.DIR_PROCESADO / "serie_por_empresa.csv", index=False)
    plt.close(fig)

    if "longitud_rama_m" in pozos:
        fig = grafico_rama_vs_acumulada(pozos)
        fig.savefig(config.DIR_FIGURAS / "02_rama_vs_acumulada_12m.png")
        plt.close(fig)
    else:
        print("  Sin datos de fractura: se omite el gráfico de rama horizontal.")
    print(f"  Figuras guardadas en {config.DIR_FIGURAS}")
