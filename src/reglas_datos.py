"""Constantes compartidas por el validador (v3) y la limpieza (apartado 5).

Tener una única fuente evita que validación y limpieza discrepen.
Corresponden a las decisiones del apartado 4 del cuaderno.
"""

# Decisión 2: población adulta.
EDAD_MINIMA = 18

# Decisión 3 y 4: recortes.
EJERCICIO_MAXIMO = 7
ALCOHOL_MINIMO = 0

# Decisión 5 (tabla 4.2): se anula el valor que cumple "valor <= limite" si
# inclusivo es True, o "valor < limite" si es False.
LIMITES_ANULACION = {
    "colesterol_hdl": (0, True),
    "trigliceridos": (0, True),
    "indice_masa_corporal": (10, False),
    "frecuencia_cardiaca_reposo": (30, False),
    "perimetro_cintura": (40, False),
}

# Decisión 6 (3.6.5 / 4.2 bis): valores artificiales repetidos.
VALORES_REPETIDOS = {
    "presion_sistolica": (196.95, 231.36),
    "colesterol_total": (371.24, 448.59),
    "trigliceridos": (357.56, 452.7),
    "acido_urico": (10.13, 12.29),
}

# Límites operativos de revisión del validador (tipo EXTREMO): (mínimo, máximo).
# None = sin límite. No son reglas de eliminación salvo las de LIMITES_ANULACION.
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
    "perimetro_cintura": (40, 250),       # v2: 50 -> 40 (apartado 4.2)
    "hemoglobina_glicosilada": (None, 20),  # v2: 4 -> sin mínimo (4.2, nota)
}

# Opciones del análisis de sensibilidad de la decisión 8.
REGLAS_LIPIDOS = ("conservar", "ldl_nan", "ambos_nan")


REGLAS_ALCOHOL = ("recortar", "nan")