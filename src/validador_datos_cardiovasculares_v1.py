from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


Gravedad = Literal["ERROR", "REVISION", "INFO"]


COLUMNAS_NUMERICAS = [
    "edad",
    "presion_sistolica",
    "presion_diastolica",
    "colesterol_total",
    "colesterol_hdl",
    "colesterol_ldl",
    "trigliceridos",
    "glucosa_ayunas",
    "indice_masa_corporal",
    "frecuencia_cardiaca_reposo",
    "horas_sueno_diarias",
    "consumo_alcohol_semanal",
    "anos_fumador",
    "proteina_c_reactiva",
    "acido_urico",
    "clearance_creatinina",
    "frecuencia_ejercicio_semanal",
    "perimetro_cintura",
    "hemoglobina_glicosilada",
]

COLUMNAS_BOOLEANAS = [
    "fumador_actual",
    "hipertension_previa",
    "diabetes_mellitus",
    "tratamiento_estatinas",
    "uso_antihipertensivos",
]

CATEGORIAS = {
    "tipo_alimentacion": [
        "Mediterránea",
        "Occidental",
        "Vegetariana",
        "Baja en carbohidratos",
    ],
    "nivel_actividad": [
        "Sedentario",
        "Ligeramente activo",
        "Moderadamente activo",
        "Muy activo",
    ],
    "antecedentes_familiares_tipo": [
        "Ninguno",
        "Materno",
        "Paterno",
        "Ambos",
    ],
}


def validar_datos_cardiovasculares(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detecta errores e inconsistencias en un dataset cardiovascular.

    No modifica el DataFrame original.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        incidencias: detalle de cada incidencia.
        resumen: número de incidencias por gravedad y regla.
    """

    columnas_requeridas = (
        COLUMNAS_NUMERICAS
        + COLUMNAS_BOOLEANAS
        + list(CATEGORIAS.keys())
    )

    # Comprobar que existen todas las columnas esperadas.
    faltantes = [
        columna
        for columna in columnas_requeridas
        if columna not in df.columns
    ]

    if faltantes:
        raise ValueError(
            f"Faltan columnas obligatorias: {faltantes}"
        )

    incidencias: list[dict] = []

    def registrar(
        mask: pd.Series,
        columna: str,
        regla: str,
        gravedad: Gravedad,
        mensaje: str,
    ) -> None:
        """Registra cada fila que incumple una regla."""

        posiciones = np.flatnonzero(mask.fillna(False).to_numpy())

        for posicion in posiciones:
            incidencias.append({
                "posicion": posicion,
                "indice_original": df.index[posicion],
                "variable": columna,
                "valor": df.iloc[posicion][columna],
                "gravedad": gravedad,
                "regla": regla,
                "mensaje": mensaje,
            })

    # Conversión numérica auxiliar, sin modificar el DataFrame.
    datos = df.copy()

    for columna in COLUMNAS_NUMERICAS:
        valores = pd.to_numeric(
            datos[columna],
            errors="coerce",
        )

        # Detectar textos que no pueden convertirse a números.
        no_numericos = (
            datos[columna].notna()
            & valores.isna()
        )

        registrar(
            no_numericos,
            columna,
            "VALOR_NO_NUMERICO",
            "ERROR",
            "El valor no puede interpretarse como número.",
        )

        # Detectar valores infinitos.
        infinitos = pd.Series(
            np.isinf(valores.to_numpy(dtype=float)),
            index=df.index,
        )

        registrar(
            infinitos,
            columna,
            "VALOR_INFINITO",
            "ERROR",
            "El valor debe ser finito.",
        )

        datos[columna] = valores.replace(
            [np.inf, -np.inf],
            np.nan,
        )

    # ---------------------------------------------------------
    # 1. Límites individuales y dominios matemáticos
    # ---------------------------------------------------------

    # Edad: población adulta.
    registrar(
        datos["edad"] < 18,
        "edad",
        "EDAD_MENOR_18",
        "ERROR",
        "El estudio está definido para población adulta.",
    )

    registrar(
        datos["edad"] > 120,
        "edad",
        "EDAD_EXTREMA",
        "REVISION",
        "Edad superior a 120 años: comprobar registro.",
    )

    registrar(
        datos["edad"].notna()
        & (datos["edad"] % 1 != 0),
        "edad",
        "EDAD_NO_ENTERA",
        "ERROR",
        "La edad debe expresarse en años cumplidos.",
    )

    # Variables que no admiten valores negativos.
    variables_no_negativas = [
        "colesterol_total",
        "colesterol_hdl",
        "colesterol_ldl",
        "trigliceridos",
        "glucosa_ayunas",
        "indice_masa_corporal",
        "consumo_alcohol_semanal",
        "anos_fumador",
        "proteina_c_reactiva",
        "acido_urico",
        "perimetro_cintura",
        "hemoglobina_glicosilada",
    ]

    for columna in variables_no_negativas:
        registrar(
            datos[columna] < 0,
            columna,
            "VALOR_NEGATIVO",
            "ERROR",
            "La variable no puede tener valores negativos.",
        )

    # Variables que requieren valores estrictamente positivos.
    variables_positivas = [
        "presion_sistolica",
        "presion_diastolica",
        "frecuencia_cardiaca_reposo",
        "clearance_creatinina",
    ]

    for columna in variables_positivas:
        registrar(
            datos[columna] <= 0,
            columna,
            "VALOR_NO_POSITIVO",
            "ERROR",
            "Se requiere un valor estrictamente positivo.",
        )

    # Horas de sueño: intervalo matemático 0-24.
    registrar(
        ~datos["horas_sueno_diarias"].between(
            0, 24, inclusive="both"
        ),
        "horas_sueno_diarias",
        "SUEÑO_FUERA_RANGO",
        "ERROR",
        "Las horas de sueño deben estar entre 0 y 24.",
    )

    # Actividad física: entre 0 y 7 días completos.
    ejercicio = datos["frecuencia_ejercicio_semanal"]

    registrar(
        ~ejercicio.between(0, 7, inclusive="both"),
        "frecuencia_ejercicio_semanal",
        "EJERCICIO_FUERA_RANGO",
        "ERROR",
        "Los días de ejercicio deben estar entre 0 y 7.",
    )

    registrar(
        ejercicio.notna() & (ejercicio % 1 != 0),
        "frecuencia_ejercicio_semanal",
        "EJERCICIO_NO_ENTERO",
        "ERROR",
        "La frecuencia semanal debe ser un número entero.",
    )

    # HbA1c: porcentaje matemáticamente limitado a 0-100.
    registrar(
        ~datos["hemoglobina_glicosilada"].between(
            0, 100, inclusive="both"
        ),
        "hemoglobina_glicosilada",
        "HBA1C_FUERA_RANGO",
        "ERROR",
        "El porcentaje de HbA1c debe estar entre 0 y 100.",
    )

    # ---------------------------------------------------------
    # 2. Relaciones cruzadas entre variables
    # ---------------------------------------------------------

    # Presión arterial: sistólica > diastólica.
    sistolica = datos["presion_sistolica"]
    diastolica = datos["presion_diastolica"]

    registrar(
        sistolica.notna()
        & diastolica.notna()
        & (sistolica <= diastolica),
        "presion_sistolica",
        "PRESION_INCOHERENTE",
        "ERROR",
        "La presión sistólica debe superar a la diastólica.",
    )

    # Perfil lipídico.
    total = datos["colesterol_total"]
    hdl = datos["colesterol_hdl"]
    ldl = datos["colesterol_ldl"]

    registrar(
        total.notna() & hdl.notna() & (hdl > total),
        "colesterol_hdl",
        "HDL_SUPERIOR_TOTAL",
        "ERROR",
        "El HDL no puede superar el colesterol total.",
    )

    registrar(
        total.notna() & ldl.notna() & (ldl > total),
        "colesterol_ldl",
        "LDL_SUPERIOR_TOTAL",
        "ERROR",
        "El LDL no puede superar el colesterol total.",
    )

    no_hdl = total - hdl

    registrar(
        ldl.notna()
        & no_hdl.notna()
        & (ldl > no_hdl),
        "colesterol_ldl",
        "LDL_SUPERIOR_NO_HDL",
        "REVISION",
        "El LDL supera el colesterol no HDL. "
        "Comprobar método y resultados analíticos.",
    )

    # Años de tabaquismo: no pueden superar la edad.
    edad = datos["edad"]
    anos_fumador = datos["anos_fumador"]

    registrar(
        edad.notna()
        & anos_fumador.notna()
        & (anos_fumador > edad),
        "anos_fumador",
        "FUMADOR_SUPERA_EDAD",
        "ERROR",
        "Los años de tabaquismo superan la edad.",
    )

    # Fumador actual con cero años de exposición.
    fumador = df["fumador_actual"]

    fumador_normalizado = (
        fumador.astype("string")
        .str.strip()
        .str.casefold()
        .isin(["1", "true", "si", "sí"])
    )

    registrar(
        fumador_normalizado
        & (anos_fumador == 0),
        "anos_fumador",
        "FUMADOR_SIN_EXPOSICION",
        "REVISION",
        "Fumador actual con cero años de exposición.",
    )

    # ---------------------------------------------------------
    # 3. Valores clínicamente extremos: NO son errores
    # ---------------------------------------------------------

    umbrales = {
        "presion_sistolica": (
            180, "gt", "Presión sistólica muy elevada."
        ),
        "presion_diastolica": (
            120, "gt", "Presión diastólica muy elevada."
        ),
        "trigliceridos": (
            400, "ge", "Triglicéridos elevados: revisar "
            "la estimación de LDL."
        ),
        "glucosa_ayunas": (
            300, "gt", "Hiperglucemia muy elevada."
        ),
        "frecuencia_cardiaca_reposo": (
            220, "gt", "Frecuencia cardíaca extremadamente elevada."
        ),
        "proteina_c_reactiva": (
            10, "gt", "PCR elevada: valorar inflamación aguda."
        ),
    }

    for columna, (umbral, operador, mensaje) in umbrales.items():

        if operador == "gt":
            mask = datos[columna] > umbral
        else:
            mask = datos[columna] >= umbral

        registrar(
            mask,
            columna,
            "VALOR_CLINICO_EXTREMO",
            "REVISION",
            mensaje,
        )

    # ---------------------------------------------------------
    # 4. Validación de variables booleanas
    # ---------------------------------------------------------

    valores_booleanos = {
        "0", "1", "true", "false", "si", "sí", "no"
    }

    for columna in COLUMNAS_BOOLEANAS:
        valores = (
            df[columna]
            .astype("string")
            .str.strip()
            .str.casefold()
        )

        invalidos = (
            df[columna].notna()
            & ~valores.isin(valores_booleanos)
        )

        registrar(
            invalidos,
            columna,
            "BOOLEANO_INVALIDO",
            "ERROR",
            "Solo se admiten 0, 1, True, False, Sí o No.",
        )

    # ---------------------------------------------------------
    # 5. Validación de categorías nominales
    # ---------------------------------------------------------

    for columna, categorias in CATEGORIAS.items():
        invalidos = (
            df[columna].notna()
            & ~df[columna].isin(categorias)
        )

        registrar(
            invalidos,
            columna,
            "CATEGORIA_NO_PERMITIDA",
            "ERROR",
            f"Categoría no válida. Valores admitidos: {categorias}",
        )

    # ---------------------------------------------------------
    # 6. Construcción del informe
    # ---------------------------------------------------------

    columnas_informe = [
        "posicion",
        "indice_original",
        "variable",
        "valor",
        "gravedad",
        "regla",
        "mensaje",
    ]

    incidencias_df = pd.DataFrame(
        incidencias,
        columns=columnas_informe,
    )

    if incidencias_df.empty:
        resumen = pd.DataFrame(
            columns=["gravedad", "regla", "incidencias"]
        )
    else:
        resumen = (
            incidencias_df
            .groupby(
                ["gravedad", "regla"],
                as_index=False,
            )
            .size()
            .rename(columns={"size": "incidencias"})
            .sort_values(
                ["gravedad", "incidencias"],
                ascending=[True, False],
            )
        )

    return incidencias_df, resumen
