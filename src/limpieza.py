"""Limpieza basada en reglas (apartado 5). Implementa las decisiones del apartado 4.

Solo contiene reglas deterministas de dominio, sin parámetros aprendidos de los
datos, así que puede aplicarse antes de dividir sin fuga de información.
Imputación, escalado, log1p y remuestreo van en el pipeline de modelado.
"""
import numpy as np
import pandas as pd

from src.reglas_datos import (
    ALCOHOL_MINIMO,
    EDAD_MINIMA,
    EJERCICIO_MAXIMO,
    LIMITES_ANULACION,
    REGLAS_ALCOHOL,
    REGLAS_LIPIDOS,
    VALORES_REPETIDOS,
)


def limpiar_datos_cardiovasculares(
    df: pd.DataFrame,
    regla_lipidos: str = "conservar",
    regla_alcohol: str = "recortar",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aplica las decisiones del apartado 4 sobre una copia de `df`.

    Parámetros
    ----------
    regla_lipidos : tratamiento de LDL > colesterol total (decisión 8):
        'conservar' (por defecto), 'ldl_nan' o 'ambos_nan'.
    regla_alcohol : tratamiento del alcohol negativo (decisión 4):
        'recortar' (a 0, por defecto) o 'nan' (se imputará en el pipeline).

    Devuelve
    --------
    df_limpio : copia limpia. Conserva el índice original (trazabilidad);
        las filas eliminadas faltan en el índice.
    registro : una fila por regla con decisión, variable, celdas y filas
        modificadas, en orden de aplicación.
    """
    if regla_lipidos not in REGLAS_LIPIDOS:
        raise ValueError(f"regla_lipidos debe ser una de {REGLAS_LIPIDOS}")
    if regla_alcohol not in REGLAS_ALCOHOL:
        raise ValueError(f"regla_alcohol debe ser una de {REGLAS_ALCOHOL}")

    out = df.copy()
    registro = []

    def anotar(decision, regla, variable, celdas, filas=None):
        registro.append({
            "decision": decision, "regla": regla, "variable": variable,
            "celdas": int(celdas),
            "filas": int(celdas if filas is None else filas),
        })

    def anular(mascara, columnas):
        """Pasa a NaN las celdas con valor que cumplen la máscara."""
        mascara = mascara.fillna(False)
        n_filas = int(mascara.sum())
        celdas = 0
        for c in columnas:
            celdas += int((mascara & out[c].notna()).sum())
            out.loc[mascara, c] = np.nan
        return celdas, n_filas

    # 1. Menores de 18 años: se eliminan las filas (decisión 2).
    menores = out["edad"] < EDAD_MINIMA
    anotar(2, "Eliminar menores de 18", "(fila)", menores.sum())
    out = out.loc[~menores]

    # 2. Recortes (decisiones 3 y 4).
    exceso = out["frecuencia_ejercicio_semanal"] > EJERCICIO_MAXIMO
    out.loc[exceso, "frecuencia_ejercicio_semanal"] = EJERCICIO_MAXIMO
    anotar(3, f"Ejercicio > {EJERCICIO_MAXIMO} recortado",
           "frecuencia_ejercicio_semanal", exceso.sum())

    neg = out["consumo_alcohol_semanal"] < ALCOHOL_MINIMO
    if regla_alcohol == "recortar":
        out.loc[neg, "consumo_alcohol_semanal"] = ALCOHOL_MINIMO
        anotar(4, "Alcohol negativo recortado a 0", "consumo_alcohol_semanal", neg.sum())
    else:
        out.loc[neg, "consumo_alcohol_semanal"] = np.nan
        anotar(4, "Alcohol negativo a NaN", "consumo_alcohol_semanal", neg.sum())

    # 3. Límites de plausibilidad a NaN (decisión 5).
    for col, (limite, inclusivo) in LIMITES_ANULACION.items():
        mascara = (out[col] <= limite) if inclusivo else (out[col] < limite)
        celdas, _ = anular(mascara, [col])
        signo = "<=" if inclusivo else "<"
        anotar(5, f"{col} {signo} {limite} a NaN", col, celdas)

    # 4. Valores repetidos a NaN (decisión 6).
    filas_afectadas = pd.Series(False, index=out.index)
    for col, valores in VALORES_REPETIDOS.items():
        mascara = pd.Series(False, index=out.index)
        for v in valores:
            mascara |= np.isclose(out[col], v, atol=0.005, rtol=0)
        filas_afectadas |= mascara
        celdas, _ = anular(mascara, [col])
        anotar(6, f"{col}: valores repetidos {valores} a NaN", col, celdas)
    anotar(6, "Total valores repetidos", "(4 variables)",
           sum(r["celdas"] for r in registro if r["decision"] == 6 and r["regla"] != "Total valores repetidos"),
           filas_afectadas.sum())

    # 5. Reglas entre variables.
    # Decisión 7: sistólica <= diastólica -> ambas a NaN.
    m = out["presion_sistolica"] <= out["presion_diastolica"]
    celdas, filas = anular(m, ["presion_sistolica", "presion_diastolica"])
    anotar(7, "Sistólica <= diastólica: ambas a NaN",
           "presion_sistolica, presion_diastolica", celdas, filas)

    # Decisión 8: LDL > colesterol total (según regla_lipidos).
    m = out["colesterol_ldl"] > out["colesterol_total"]
    if regla_lipidos == "conservar":
        anotar(8, "LDL > total: conservado (registrado)", "colesterol_ldl",
               0, m.sum())
    else:
        columnas = (["colesterol_ldl"] if regla_lipidos == "ldl_nan"
                    else ["colesterol_ldl", "colesterol_total"])
        celdas, filas = anular(m, columnas)
        anotar(8, f"LDL > total: {regla_lipidos}", ", ".join(columnas), celdas, filas)

    # Decisión 9: años de fumador > edad -> NaN.
    m = out["anos_fumador"] > out["edad"]
    celdas, _ = anular(m, ["anos_fumador"])
    anotar(9, "Años fumador > edad a NaN", "anos_fumador", celdas)

    registro = pd.DataFrame(registro)
    return out, registro
