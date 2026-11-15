# Registro de decisiones

En este documento incluimos las decisiones más relevantes tomadas a lo largo del proyecto. Cada decisión está desarrollada en el cuaderno `notebooks/Tarea1_ML.ipynb`; aquí resumimos lo que se ha decidido, por qué y dónde se aplica.

## 1. Ideas generales

- **En el EDA se diagnostica, no se corrige.** El análisis exploratorio (apartado 3) mira el dataset completo, pero no aplica sobre él ninguna transformación que aprenda parámetros.
- **Reglas deterministas frente a transformaciones aprendidas.**
  - La limpieza (apartados 4 y 5) usa reglas con umbrales fijos, que se aplican igual a un paciente nuevo.
  - Las transformaciones que aprenden de los datos (imputación, escalado, codificación, selección de variables y SMOTE) se ajustan solo con el conjunto de entrenamiento (apartado 6).
- **Reproducibilidad.** Semilla 42 en todo lo aleatorio, versiones de las librerías fijadas en requirements.txt y Python 3.12.

## 2. Hallazgos del EDA que motivan las decisiones

| Problemática | Alcance | Evidencia |
|---|---|---|
| Valores ausentes | LDL, glucosa, sueño y PCR (5–8 %); 533 filas (23,6 %) | Sin asociación significativa con el riesgo; MCAR como hipótesis de trabajo, no demostrada |
| Valores negativos | HDL 4, triglicéridos 4, alcohol 177 | En el alcohol son la cola baja de la variable (146 de 177 de riesgo bajo) |
| Valores implausibles | IMC de 3,6 a 9,7; frecuencia cardíaca de 25,8 y 26,6 | Incompatibles con una medición en un adulto vivo |
| Valores repetidos | 2 valores fijos en sistólica, colesterol total, triglicéridos y ácido úrico; 420 filas | Situados a 3,5 y 5,25 desviaciones típicas del resto; sin asociación con el riesgo; generan toda la asimetría |
| Edad y ejercicio | 4 menores de 18; 41 ejercicios > 7; 91 % y 81 % con decimales | Población adulta; la semana tiene 7 días |
| Incoherencias entre variables | Sistólica ≤ diastólica 44; LDL > total 140; LDL > no-HDL 559 | Número próximo al esperable si las variables fueran independientes (≈ 56, ≈ 156, ≈ 565) |
| Desbalanceo | 50 / 30 / 20 (2,5 : 1) | Clasificador trivial: exactitud 0,50 |

## 3. Decisiones de limpieza (apartado 4)

| # | Problema | Decisión | Afectados |
|---|---|---|---|
| 1 | Edad con decimales | Aceptarla como variable continua | — |
| 2 | Menores de 18 años | Eliminar las filas | 4 filas |
| 3 | Ejercicio > 7 días | Aceptar decimales y recortar a 7 | 41 valores |
| 4 | Alcohol negativo | Recortar a 0 como decisión operativa; alternativa NaN pendiente de evaluar | 177 valores |
| 5 | Valores imposibles o implausibles | Convertir el valor en NaN (tabla de límites) | 18 valores |
| 6 | Valores repetidos en cuatro variables | Convertir en NaN solo esos valores | 448 celdas en 419 pacientes |
| 7 | Sistólica ≤ diastólica | Ambas presiones a NaN | 44 filas |
| 8 | LDL > colesterol total | Conservar y registrar; alternativas (LDL a NaN, ambos a NaN) en el análisis de sensibilidad | 140 filas |
| 9 | Años de fumador > edad | Años de fumador a NaN | 0 (el único caso era un menor) |

**Límites de plausibilidad (decisión 5).** Se anula el valor, no al paciente.

| Variable | No plausible | Afectados |
|---|---|---|
| `colesterol_hdl` | ≤ 0 | 4 |
| `trigliceridos` | ≤ 0 | 4 |
| `indice_masa_corporal` | < 10 kg/m² | 8 |
| `frecuencia_cardiaca_reposo` | < 30 lpm | 2 |
| `perimetro_cintura` | < 40 cm | 0 (regla para datos nuevos) |

La HbA1c no tiene límite de anulación: los valores bajos se conservan.

### Justificaciones clave

- **Decisión 4 (alcohol).**
  - Los negativos son inválidos, pero no se conoce el valor real: recortar a 0 es una asunción.
  - Se prefiere al NaN porque imputar con la mediana (4,3 UBE, la de los valores no negativos) los llevaría al centro de la escala.
  - Crea un pico de 184 ceros. El alcohol es la variable más asociada al riesgo (ρ = 0,39), por eso conviene evaluar la alternativa NaN (pendiente).
- **Decisión 6 (valores repetidos).**
  - Las evidencias apuntan a valores generados o sustituidos artificialmente.
  - Se tratan como datos ausentes, asumiendo que podrían ocultar valores altos (posible MNAR). Se imputan con la mediana, sin indicadores de ausencia. El posible sesgo MNAR queda como limitación.
- **Decisiones 7 y 8 (mismo principio, distinto tratamiento).**
  - En ambos casos no se puede saber cuál de las dos variables es errónea.
  - La presión se anula porque afecta al 2 % de las filas e incluye valores absurdos por sí mismos (sistólicas de 52–63 mmHg con diastólicas de unos 90).
  - LDL > total afecta al 6,2 %, y anularlo duplicaría los ausentes del LDL (del 6 % al 12 %). Cada valor lipídico por separado es plausible. Por eso se conserva. Queda pendiente comprobar si anularlo cambia los resultados.

## 4. Resultado de la limpieza (apartado 5)

- De 2254 a 2250 filas.
- 218 valores recortados.
- Filas con algún NaN: del 23,6 % al 39,2 %.
- Validador v3 sobre los datos limpios: 0 incidencias con decisión `corregir`. Quedan 698 `conservar` (LDL > total y LDL > no-HDL) y 109 `revision`.
- El CSV limpio se guarda en `data/processed/enfermedad_cardiovascular_limpio.csv`.

## 5. Decisiones de preprocesamiento (apartado 6)

Todas las transformaciones de este apartado aprenden de los datos, por lo que se ajustan solo con el conjunto de entrenamiento y se aplican después al de test.

| Paso | Decisión | Motivo |
|---|---|---|
| Partición | 80 / 20 estratificada (semilla 42) | Conservar el reparto 50 / 30 / 20 en entrenamiento y test |
| Imputación | Mediana | Los ausentes originales son compatibles con MCAR; la mediana es robusta frente a la asimetría |
| Codificación | One-hot N−1 en `tipo_alimentacion` y `antecedentes_familiares_tipo`; ordinal 0–3 en `nivel_actividad` | Las dos primeras no tienen orden; `nivel_actividad` sí lo tiene |
| Escalado | `StandardScaler` en las variables no binarias | Las variables tienen escalas muy distintas, lo que afecta a la regresión logística y a k-NN |
| Selección de variables | `SelectKBest(k=18)` | Reducir las variables con poca relación con el riesgo |
| Desbalanceo | SMOTE hasta el 65 % de la clase mayoritaria; NCL descartado | NCL reducía la sensibilidad del riesgo alto |

## 6. Aclaraciones y aspectos pendientes

- **Análisis de sensibilidad de las decisiones 4 y 8.** No se ha realizado. Las alternativas (alcohol negativo a NaN; LDL o ambas variables lipídicas a NaN) quedan como trabajo pendiente.
- **Indicadores de ausencia.** No se han añadido. El posible sesgo MNAR de los valores repetidos convertidos en NaN (decisión 6) queda como limitación.
- **Versiones de las librerías.** Están fijadas en `requirements.txt`.
- **CSV limpio.** Se guarda en `data/processed/enfermedad_cardiovascular_limpio.csv` sin la columna `id_original`.
- **Limitación del dataset.** Faltan asociaciones fisiológicas esperables: presión sistólica y diastólica, IMC y cintura, glucosa y HbA1c, colesterol total y LDL (ρ ≈ 0 en todos los casos). Las incoherencias entre variables aparecen en un número próximo al esperable por azar. El rendimiento de los modelos puede no trasladarse a pacientes reales.
