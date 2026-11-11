
import numpy as np
import pandas as pd


def validar_datos_cardiovasculares(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Valida las variables del estudio cardiovascular.

    Devuelve:
        incidencias: detalle de cada anomalía detectada.
        resumen: número de incidencias por variable, tipo y regla.

    Tipos de incidencia:
        INVALIDO: incumplimiento de dominio o formato.
        INCONSISTENCIA: relación entre variables que requiere revisión.
        EXTREMO: fuera de límites operativos de plausibilidad.

    Los umbrales EXTREMO no son límites fisiológicos absolutos.
    La función no modifica el DataFrame original.
    """

    # ========================================================
    # 1. CONFIGURACIÓN
    # ========================================================

    # Umbrales operativos de revisión: mínimo y máximo.
    LIMITES_PLAUSIBILIDAD = {
        "edad": (18, 110),
        "presion_sistolica": (50, 300),
        "presion_diastolica": (20, 180),
        "colesterol_total": (50, 1000),
        "colesterol_hdl": (10, 150),
        "colesterol_ldl": (10, 700),
        "trigliceridos": (10, 5000),
        "glucosa_ayunas": (40, 1000),
        "indice_masa_corporal": (10, 100),
        "frecuencia_cardiaca_reposo": (30, 220),
        "horas_sueno_diarias": (2, 16),
        "consumo_alcohol_semanal": (0, 100),
        "anos_fumador": (0, 80),
        "proteina_c_reactiva": (0, 300),
        "acido_urico": (2, 25),
        "clearance_creatinina": (5, 200),
        "frecuencia_ejercicio_semanal": (0, 7),
        "perimetro_cintura": (50, 250),
        "hemoglobina_glicosilada": (4, 20),
    }

    # Valores permitidos para las variables nominales.
    CATEGORIAS = {
        "tipo_alimentacion": {
            "Mediterránea",
            "Occidental",
            "Vegetariana",
            "Baja en carbohidratos",
        },
        "nivel_actividad": {
            "Sedentario",
            "Ligeramente activo",
            "Moderadamente activo",
            "Muy activo",
        },
        "antecedentes_familiares_tipo": {
            "Ninguno",
            "Materno",
            "Paterno",
            "Ambos",
        },
    }

    BOOLEANAS = [
        "fumador_actual",
        "hipertension_previa",
        "diabetes_mellitus",
        "tratamiento_estatinas",
        "uso_antihipertensivos",
    ]

    # Variables que no pueden ser negativas.
    NO_NEGATIVAS = [
        "presion_diastolica",
        "colesterol_total",
        "colesterol_hdl",
        "colesterol_ldl",
        "trigliceridos",
        "glucosa_ayunas",
        "horas_sueno_diarias",
        "consumo_alcohol_semanal",
        "anos_fumador",
        "proteina_c_reactiva",
        "acido_urico",
        "clearance_creatinina",
        "frecuencia_ejercicio_semanal",
        "hemoglobina_glicosilada",
    ]

    # Variables que deben ser estrictamente positivas.
    POSITIVAS = [
        "presion_sistolica",
        "indice_masa_corporal",
        "frecuencia_cardiaca_reposo",
        "perimetro_cintura",
    ]

    COLUMNAS_RESULTADO = [
        "indice",
        "variable",
        "valor",
        "tipo",
        "regla",
        "detalle",
    ]

    incidencias = []

    # ========================================================
    # 2. PREPARACIÓN
    # ========================================================

    if not df.index.is_unique:
        raise ValueError(
            "El índice del DataFrame debe ser único. "
            "Utiliza reset_index(drop=True) si procede."
        )

    columnas_numericas = list(LIMITES_PLAUSIBILIDAD)

    columnas_esperadas = (
        columnas_numericas
        + BOOLEANAS
        + list(CATEGORIAS)
    )

    def registrar(
        variable: str,
        mascara: pd.Series,
        tipo: str,
        regla: str,
        detalle: str,
    ) -> None:
        """Registra las filas que incumplen una regla."""
        mascara = mascara.fillna(False).astype(bool)

        for indice in df.index[mascara]:
            incidencias.append({
                "indice": indice,
                "variable": variable,
                "valor": (
                    df.at[indice, variable]
                    if variable in df.columns
                    else np.nan
                ),
                "tipo": tipo,
                "regla": regla,
                "detalle": detalle,
            })

    # Columnas ausentes
    for columna in columnas_esperadas:
        if columna not in df.columns:
            incidencias.append({
                "indice": None,
                "variable": columna,
                "valor": np.nan,
                "tipo": "INVALIDO",
                "regla": "Columna ausente",
                "detalle": "La columna no existe en el DataFrame.",
            })

    # Solo se procesan columnas numéricas disponibles.
    numericas_disponibles = [
        col for col in columnas_numericas
        if col in df.columns
    ]

    numericos = df[numericas_disponibles].apply(
        pd.to_numeric,
        errors="coerce",
    )

    # ========================================================
    # 3. VALIDACIÓN DE TIPOS Y VALORES AUSENTES
    # ========================================================

    for columna in numericas_disponibles:
        original = df[columna]
        valores = numericos[columna]

        registrar(
            columna,
            original.notna() & valores.isna(),
            "INVALIDO",
            "Valor no numérico",
            "No puede convertirse a un número.",
        )

        registrar(
            columna,
            original.isna(),
            "INVALIDO",
            "Valor ausente",
            "El registro no contiene un valor.",
        )

        registrar(
            columna,
            pd.Series(
                np.isinf(valores.to_numpy()),
                index=df.index,
            ),
            "INVALIDO",
            "Valor infinito",
            "No se admite infinito positivo o negativo.",
        )

    # Sustituir infinitos para las comprobaciones posteriores.
    numericos = numericos.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # ========================================================
    # 4. RESTRICCIONES DE DOMINIO
    # ========================================================

    for columna in POSITIVAS:
        if columna in numericos:
            registrar(
                columna,
                numericos[columna] <= 0,
                "INVALIDO",
                "Valor no positivo",
                "La variable debe ser mayor que cero.",
            )

    for columna in NO_NEGATIVAS:
        if columna in numericos:
            registrar(
                columna,
                numericos[columna] < 0,
                "INVALIDO",
                "Valor negativo",
                "La variable no admite valores negativos.",
            )

    # Edad: estudio limitado a población adulta.
    if "edad" in numericos:
        edad = numericos["edad"]

        registrar(
            "edad",
            edad < 18,
            "INVALIDO",
            "Edad inferior a 18",
            "Fuera de la población definida para el estudio.",
        )

        registrar(
            "edad",
            edad.notna() & (edad % 1 != 0),
            "INVALIDO",
            "Edad no entera",
            "La edad se registra en años completos.",
        )

    # Ejercicio: días semanales.
    if "frecuencia_ejercicio_semanal" in numericos:
        ejercicio = numericos["frecuencia_ejercicio_semanal"]

        registrar(
            "frecuencia_ejercicio_semanal",
            ejercicio > 7,
            "INVALIDO",
            "Más de 7 días semanales",
            "Una semana tiene siete días.",
        )

        registrar(
            "frecuencia_ejercicio_semanal",
            ejercicio.notna() & (ejercicio % 1 != 0),
            "INVALIDO",
            "Número de días no entero",
            "La frecuencia se expresa en días completos.",
        )

    # Sueño: horas por día.
    if "horas_sueno_diarias" in numericos:
        registrar(
            "horas_sueno_diarias",
            numericos["horas_sueno_diarias"] > 24,
            "INVALIDO",
            "Más de 24 horas diarias",
            "El día tiene 24 horas.",
        )

    # HbA1c expresada como porcentaje.
    if "hemoglobina_glicosilada" in numericos:
        registrar(
            "hemoglobina_glicosilada",
            numericos["hemoglobina_glicosilada"] > 100,
            "INVALIDO",
            "Porcentaje superior a 100",
            "La variable está expresada en porcentaje.",
        )

    # ========================================================
    # 5. LÍMITES INFERIORES Y SUPERIORES DE PLAUSIBILIDAD
    # ========================================================

    for columna, (minimo, maximo) in LIMITES_PLAUSIBILIDAD.items():
        if columna not in numericos:
            continue

        valores = numericos[columna]

        registrar(
            columna,
            valores < minimo,
            "EXTREMO",
            "Inferior al límite de plausibilidad",
            f"Umbral inferior operativo: {minimo}.",
        )

        registrar(
            columna,
            valores > maximo,
            "EXTREMO",
            "Superior al límite de plausibilidad",
            f"Umbral superior operativo: {maximo}.",
        )

    # ========================================================
    # 6. INCONSISTENCIAS ENTRE VARIABLES
    # ========================================================

    def comparar(
        columna_a: str,
        columna_b: str,
        condicion,
        regla: str,
        detalle: str,
    ) -> None:
        """Comprueba una relación entre dos columnas numéricas."""
        if columna_a not in numericos or columna_b not in numericos:
            return

        mascara = condicion(
            numericos[columna_a],
            numericos[columna_b],
        )

        registrar(
            columna_a,
            mascara,
            "INCONSISTENCIA",
            regla,
            detalle,
        )

    comparar(
        "presion_sistolica",
        "presion_diastolica",
        lambda a, b: a <= b,
        "Sistólica <= diastólica",
        "Revisar la relación entre ambas presiones.",
    )

    comparar(
        "colesterol_hdl",
        "colesterol_total",
        lambda a, b: a > b,
        "HDL > colesterol total",
        "El HDL no debería superar el colesterol total.",
    )

    comparar(
        "colesterol_ldl",
        "colesterol_total",
        lambda a, b: a > b,
        "LDL > colesterol total",
        "El LDL no debería superar el colesterol total.",
    )

    comparar(
        "anos_fumador",
        "edad",
        lambda a, b: a > b,
        "Años fumando > edad",
        "Los años como fumador no pueden superar la edad.",
    )

    # Relación adicional del perfil lipídico.
    if all(
        col in numericos
        for col in [
            "colesterol_total",
            "colesterol_hdl",
            "colesterol_ldl",
        ]
    ):
        colesterol_no_hdl = (
            numericos["colesterol_total"]
            - numericos["colesterol_hdl"]
        )

        registrar(
            "colesterol_ldl",
            numericos["colesterol_ldl"] > colesterol_no_hdl,
            "INCONSISTENCIA",
            "LDL > colesterol no-HDL",
            "Revisar unidades, métodos y coherencia del perfil lipídico.",
        )

    # ========================================================
    # 7. ALERTAS CLÍNICAS ESPECÍFICAS
    # ========================================================

    alertas_clinicas = {
        "presion_sistolica": [
            (lambda s: s > 180, "Sistólica > 180 mmHg"),
        ],
        "presion_diastolica": [
            (lambda s: s > 120, "Diastólica > 120 mmHg"),
        ],
        "glucosa_ayunas": [
            (lambda s: s < 54, "Glucosa < 54 mg/dL"),
            (lambda s: s > 300, "Glucosa > 300 mg/dL"),
        ],
        "trigliceridos": [
            (lambda s: s >= 500, "Triglicéridos >= 500 mg/dL"),
            (lambda s: s >= 1000, "Triglicéridos >= 1000 mg/dL"),
        ],
        "proteina_c_reactiva": [
            (lambda s: s > 10, "PCR > 10 mg/L"),
        ],
        "hemoglobina_glicosilada": [
            (lambda s: s < 4, "HbA1c < 4 %"),
            (lambda s: s > 15, "HbA1c > 15 %"),
        ],
    }

    for columna, reglas in alertas_clinicas.items():
        if columna not in numericos:
            continue

        for condicion, descripcion in reglas:
            registrar(
                columna,
                condicion(numericos[columna]),
                "ALERTA_CLINICA",
                descripcion,
                "Valor clínicamente relevante; requiere revisión.",
            )

    # ========================================================
    # 8. VALIDACIÓN DE VARIABLES BOOLEANAS
    # ========================================================

    for columna in BOOLEANAS:
        if columna not in df.columns:
            continue

        registrar(
            columna,
            df[columna].isna(),
            "INVALIDO",
            "Valor ausente",
            "La variable booleana no tiene valor.",
        )

        registrar(
            columna,
            df[columna].notna() &
            ~df[columna].isin([0, 1, True, False]),
            "INVALIDO",
            "Valor booleano no permitido",
            "Se admiten 0/1 o True/False.",
        )

    # ========================================================
    # 9. VALIDACIÓN DE VARIABLES CATEGÓRICAS
    # ========================================================

    for columna, categorias_permitidas in CATEGORIAS.items():
        if columna not in df.columns:
            continue

        registrar(
            columna,
            df[columna].isna(),
            "INVALIDO",
            "Valor ausente",
            "La variable categórica no tiene valor.",
        )

        registrar(
            columna,
            df[columna].notna() &
            ~df[columna].isin(categorias_permitidas),
            "INVALIDO",
            "Categoría no permitida",
            f"Categorías válidas: {sorted(categorias_permitidas)}",
        )

    # ========================================================
    # 10. RESULTADOS
    # ========================================================

    tabla_incidencias = pd.DataFrame(
        incidencias,
        columns=COLUMNAS_RESULTADO,
    )

    if tabla_incidencias.empty:
        tabla_resumen = pd.DataFrame(
            columns=[
                "variable",
                "tipo",
                "regla",
                "casos",
            ]
        )
    else:
        tabla_resumen = (
            tabla_incidencias
            .groupby(
                ["variable", "tipo", "regla"],
                dropna=False,
            )
            .size()
            .reset_index(name="casos")
            .sort_values(
                ["casos", "variable"],
                ascending=[False, True],
            )
            .reset_index(drop=True)
        )

    return tabla_incidencias, tabla_resumen
