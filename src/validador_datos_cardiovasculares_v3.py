import numpy as np
import pandas as pd

from src.reglas_datos import (
    EDAD_MINIMA,
    EJERCICIO_MAXIMO,
    LIMITES_ANULACION,
    LIMITES_PLAUSIBILIDAD,
    VALORES_REPETIDOS,
)

BOOLEANAS = [
    "fumador_actual",
    "hipertension_previa",
    "diabetes_mellitus",
    "tratamiento_estatinas",
    "uso_antihipertensivos",
]

CATEGORIAS = {
    "tipo_alimentacion": {
        "Mediterránea", "Occidental", "Vegetariana", "Baja en carbohidratos",
    },
    "nivel_actividad": {
        "Sedentario", "Ligeramente activo", "Moderadamente activo", "Muy activo",
    },
    "antecedentes_familiares_tipo": {"Ninguno", "Materno", "Paterno", "Ambos"},
}

COLUMNAS_RESULTADO = [
    "indice", "variable", "valor", "tipo", "regla", "decision", "detalle",
]


def _decision(variable: str, tipo: str, regla: str) -> str:
    """Tratamiento previsto por el apartado 4 para cada incidencia.

    corregir  -> la limpieza la elimina; tras limpiar no debe aparecer.
    conservar -> se mantiene a propósito y se registra.
    revision  -> aviso informativo; no se modifica.
    """
    if regla in {
        "Edad inferior a 18",
        "Más de 7 días semanales",
        "Valor repetido anómalo",
        "Sistólica <= diastólica",
        "Años fumando > edad",
        "Valor negativo",
        "Valor no positivo",
    }:
        return "corregir"
    if tipo == "EXTREMO" and regla == "Inferior al límite de plausibilidad":
        # Solo si el umbral del validador coincide con el de anulación (4.2).
        # HDL y TG se anulan con <= 0 (INVALIDO), no con el umbral de revisión.
        limite = LIMITES_ANULACION.get(variable)
        minimo = LIMITES_PLAUSIBILIDAD[variable][0]
        coincide = limite is not None and not limite[1] and limite[0] == minimo
        return "corregir" if coincide else "revision"
    if regla in {"LDL > colesterol total", "LDL > colesterol no-HDL"}:
        return "conservar"
    return "revision"


def validar_datos_cardiovasculares(
    df: pd.DataFrame,
    exigir_enteros: bool = False,
    incluir_ausentes: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Valida las variables del estudio cardiovascular (v3).

    Parámetros
    ----------
    exigir_enteros : si True, marca edad y días de ejercicio no enteros.
        Por defecto False: las decisiones 1 y 3 aceptan decimales.
    incluir_ausentes : si True, añade los valores ausentes como incidencias
        de tipo AUSENTE. Por defecto False: los NaN originales se
        imputan en el pipeline y se cuentan con `resumen_ausentes`.

    Devuelve
    --------
    incidencias : una fila por anomalía, con la columna `decision`
        (corregir / conservar / revision, ver `_decision`).
    resumen : número de casos por variable, tipo, regla y decisión.

    Tipos: INVALIDO, INCONSISTENCIA, EXTREMO, ALERTA_CLINICA, ARTEFACTO,
    AUSENTE (opcional). No modifica el DataFrame original.
    """
    if not df.index.is_unique:
        raise ValueError(
            "El índice del DataFrame debe ser único. "
            "Utiliza reset_index(drop=True) si procede."
        )

    columnas_numericas = list(LIMITES_PLAUSIBILIDAD)
    columnas_esperadas = columnas_numericas + BOOLEANAS + list(CATEGORIAS)

    no_negativas = [
        "presion_diastolica", "colesterol_total", "colesterol_hdl",
        "colesterol_ldl", "trigliceridos", "glucosa_ayunas",
        "horas_sueno_diarias", "consumo_alcohol_semanal", "anos_fumador",
        "proteina_c_reactiva", "acido_urico", "clearance_creatinina",
        "frecuencia_ejercicio_semanal", "hemoglobina_glicosilada",
    ]
    positivas = [
        "presion_sistolica", "indice_masa_corporal",
        "frecuencia_cardiaca_reposo", "perimetro_cintura",
    ]

    incidencias = []

    def registrar(variable, mascara, tipo, regla, detalle):
        mascara = mascara.fillna(False).astype(bool)
        for indice in df.index[mascara]:
            incidencias.append({
                "indice": indice,
                "variable": variable,
                "valor": df.at[indice, variable] if variable in df.columns else np.nan,
                "tipo": tipo,
                "regla": regla,
                "decision": _decision(variable, tipo, regla),
                "detalle": detalle,
            })

    for columna in columnas_esperadas:
        if columna not in df.columns:
            incidencias.append({
                "indice": None, "variable": columna, "valor": np.nan,
                "tipo": "INVALIDO", "regla": "Columna ausente",
                "decision": "revision",
                "detalle": "La columna no existe en el DataFrame.",
            })

    disponibles = [c for c in columnas_numericas if c in df.columns]
    numericos = df[disponibles].apply(pd.to_numeric, errors="coerce")

    # --- Tipos y ausentes -------------------------------------------------
    for columna in disponibles:
        original, valores = df[columna], numericos[columna]
        registrar(columna, original.notna() & valores.isna(), "INVALIDO",
                  "Valor no numérico", "No puede convertirse a un número.")
        registrar(columna,
                  pd.Series(np.isinf(valores.to_numpy()), index=df.index),
                  "INVALIDO", "Valor infinito",
                  "No se admite infinito positivo o negativo.")
        if incluir_ausentes:
            registrar(columna, original.isna(), "AUSENTE", "Valor ausente",
                      "El registro no contiene un valor (se imputa en el pipeline).")

    numericos = numericos.replace([np.inf, -np.inf], np.nan)

    # --- Dominio ----------------------------------------------------------
    for columna in positivas:
        if columna in numericos:
            registrar(columna, numericos[columna] <= 0, "INVALIDO",
                      "Valor no positivo", "La variable debe ser mayor que cero.")
    for columna in no_negativas:
        if columna in numericos:
            registrar(columna, numericos[columna] < 0, "INVALIDO",
                      "Valor negativo", "La variable no admite valores negativos.")

    if "edad" in numericos:
        edad = numericos["edad"]
        registrar("edad", edad < EDAD_MINIMA, "INVALIDO", "Edad inferior a 18",
                  "Fuera de la población definida para el estudio.")
        if exigir_enteros:
            registrar("edad", edad.notna() & (edad % 1 != 0), "INVALIDO",
                      "Edad no entera", "La edad se registra en años completos.")

    if "frecuencia_ejercicio_semanal" in numericos:
        ej = numericos["frecuencia_ejercicio_semanal"]
        registrar("frecuencia_ejercicio_semanal", ej > EJERCICIO_MAXIMO,
                  "INVALIDO", "Más de 7 días semanales",
                  "Una semana tiene siete días.")
        if exigir_enteros:
            registrar("frecuencia_ejercicio_semanal",
                      ej.notna() & (ej % 1 != 0), "INVALIDO",
                      "Número de días no entero",
                      "La frecuencia se expresa en días completos.")

    if "horas_sueno_diarias" in numericos:
        registrar("horas_sueno_diarias", numericos["horas_sueno_diarias"] > 24,
                  "INVALIDO", "Más de 24 horas diarias", "El día tiene 24 horas.")

    if "hemoglobina_glicosilada" in numericos:
        registrar("hemoglobina_glicosilada",
                  numericos["hemoglobina_glicosilada"] > 100, "INVALIDO",
                  "Porcentaje superior a 100",
                  "La variable está expresada en porcentaje.")

    # --- Límites de plausibilidad ----------------------------------------
    for columna, (minimo, maximo) in LIMITES_PLAUSIBILIDAD.items():
        if columna not in numericos:
            continue
        if minimo is not None:
            registrar(columna, numericos[columna] < minimo, "EXTREMO",
                      "Inferior al límite de plausibilidad",
                      f"Umbral inferior operativo: {minimo}.")
        if maximo is not None:
            registrar(columna, numericos[columna] > maximo, "EXTREMO",
                      "Superior al límite de plausibilidad",
                      f"Umbral superior operativo: {maximo}.")

    # --- Valores artificiales repetidos (decisión 6) ------------------------
    for columna, valores_rep in VALORES_REPETIDOS.items():
        if columna in numericos:
            mascara = pd.Series(False, index=df.index)
            for v in valores_rep:
                mascara |= np.isclose(numericos[columna], v, atol=0.005)
            registrar(columna, mascara, "ARTEFACTO", "Valor repetido anómalo",
                      f"Valores artificiales repetidos {valores_rep} (apartado 4.2 bis).")

    # --- Inconsistencias entre variables ---------------------------------
    def comparar(a, b, condicion, regla, detalle):
        if a in numericos and b in numericos:
            registrar(a, condicion(numericos[a], numericos[b]),
                      "INCONSISTENCIA", regla, detalle)

    comparar("presion_sistolica", "presion_diastolica", lambda a, b: a <= b,
             "Sistólica <= diastólica", "Revisar la relación entre ambas presiones.")
    comparar("colesterol_hdl", "colesterol_total", lambda a, b: a > b,
             "HDL > colesterol total", "El HDL no debería superar el colesterol total.")
    comparar("colesterol_ldl", "colesterol_total", lambda a, b: a > b,
             "LDL > colesterol total",
             "El LDL no debería superar el total. Se conserva y se evalúa con análisis de sensibilidad (decisión 8).")
    comparar("anos_fumador", "edad", lambda a, b: a > b,
             "Años fumando > edad", "Los años como fumador no pueden superar la edad.")

    if all(c in numericos for c in
           ["colesterol_total", "colesterol_hdl", "colesterol_ldl"]):
        no_hdl = numericos["colesterol_total"] - numericos["colesterol_hdl"]
        registrar("colesterol_ldl", numericos["colesterol_ldl"] > no_hdl,
                  "INCONSISTENCIA", "LDL > colesterol no-HDL",
                  "Revisar coherencia del perfil lipídico. Se conserva (decisión 8).")

    # --- Alertas clínicas -----------------------------------------------
    alertas = {
        "presion_sistolica": [(lambda s: s > 180, "Sistólica > 180 mmHg")],
        "presion_diastolica": [(lambda s: s > 120, "Diastólica > 120 mmHg")],
        "glucosa_ayunas": [(lambda s: s < 54, "Glucosa < 54 mg/dL"),
                           (lambda s: s > 300, "Glucosa > 300 mg/dL")],
        "trigliceridos": [(lambda s: s >= 500, "Triglicéridos >= 500 mg/dL"),
                          (lambda s: s >= 1000, "Triglicéridos >= 1000 mg/dL")],
        "proteina_c_reactiva": [(lambda s: s > 10, "PCR > 10 mg/L")],
        "hemoglobina_glicosilada": [(lambda s: s < 4, "HbA1c < 4 %"),
                                    (lambda s: s > 15, "HbA1c > 15 %")],
    }
    for columna, reglas in alertas.items():
        if columna in numericos:
            for condicion, descripcion in reglas:
                registrar(columna, condicion(numericos[columna]),
                          "ALERTA_CLINICA", descripcion,
                          "Valor clínicamente relevante; requiere revisión.")

    # --- Booleanas y categóricas -----------------------------------------
    for columna in BOOLEANAS:
        if columna in df.columns:
            if incluir_ausentes:
                registrar(columna, df[columna].isna(), "AUSENTE",
                          "Valor ausente", "La variable booleana no tiene valor.")
            registrar(columna,
                      df[columna].notna() & ~df[columna].isin([0, 1, True, False]),
                      "INVALIDO", "Valor booleano no permitido",
                      "Se admiten 0/1 o True/False.")

    for columna, permitidas in CATEGORIAS.items():
        if columna in df.columns:
            if incluir_ausentes:
                registrar(columna, df[columna].isna(), "AUSENTE",
                          "Valor ausente", "La variable categórica no tiene valor.")
            registrar(columna,
                      df[columna].notna() & ~df[columna].isin(permitidas),
                      "INVALIDO", "Categoría no permitida",
                      f"Categorías válidas: {sorted(permitidas)}")

    tabla = pd.DataFrame(incidencias, columns=COLUMNAS_RESULTADO)
    if tabla.empty:
        resumen = pd.DataFrame(
            columns=["variable", "tipo", "regla", "decision", "casos"])
    else:
        resumen = (tabla.groupby(["variable", "tipo", "regla", "decision"],
                                 dropna=False)
                   .size().reset_index(name="casos")
                   .sort_values(["casos", "variable"], ascending=[False, True])
                   .reset_index(drop=True))
    return tabla, resumen


def resumen_ausentes(df: pd.DataFrame) -> pd.DataFrame:
    """Valores ausentes por columna (número y porcentaje), de mayor a menor."""
    n = df.isna().sum()
    out = pd.DataFrame({"ausentes": n, "porcentaje": (100 * n / len(df)).round(2)})
    return out[out["ausentes"] > 0].sort_values("ausentes", ascending=False)


def verificar_limpieza(df_limpio: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Valida un DataFrame ya limpio y devuelve el resumen por decisión.

    La limpieza es correcta si no queda ninguna incidencia con decisión
    'corregir'. Las de 'conservar' y 'revision' deben permanecer.
    """
    _, resumen = validar_datos_cardiovasculares(df_limpio, **kwargs)
    return resumen
