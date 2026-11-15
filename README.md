# Machine Learning Práctica 1
## Predicción Multiclase de Riesgo Cardiovascular en Población Adulta

Proyecto de aprendizaje supervisado que clasifica a pacientes adultos en tres niveles de riesgo cardiovascular (Clase 0: Bajo Riesgo, Clase 1: Riesgo Moderado y Clase 2: Alto Riesgo) a partir de 28 variables sociodemográficas, clínicas, analíticas y de estilo de vida. El proyecto ha sido planteado por el equipo médico e investigador del Hospital Morales Meseguer, que aporta una cohorte clínica con 2.254 registros de pacientes.

## Equipo

| Miembro | Usuario de GitHub |
|---|---|
| Lucas Morales Chacón | @lmc08840 |
| Carlos Sánchez Maartín | @csm8-um |


## Estructura del proyecto

```
├── notebooks/Tarea1_ML.ipynb      # cuaderno de la práctica 1
├── src/
│   ├── reglas_datos.py            # constantes compartidas por la limpieza y el validador
│   ├── limpieza.py                # limpieza basada en reglas (apartado 5)
│   └── validador_datos_cardiovasculares_v1.py / _v2.py / _v3.py
├── data/raw/                      # dataset original
├── data/processed/                # dataset limpio
├── reports/generated/             # informes del validador, generados por el cuaderno
└── docs/                          # registro de decisiones y de uso de IA
```


## Reproducir el proyecto

Requisitos: **Python 3.12** y git.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (PowerShell)
# source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

Abrir `notebooks/Tarea1_ML.ipynb`, seleccionar el intérprete de `.venv` y ejecutar **Restart Kernel and Run All**. El cuaderno imprime al inicio las versiones de Python y de las librerías, y fija la semilla 42 en todo lo aleatorio.

## Documentación

- [`docs/decisiones.md`](docs/decisiones.md): decisiones de diseño y su justificación.
- [`docs/uso_ia.md`](docs/uso_ia.md): registro de uso de herramientas de IA generativa.
