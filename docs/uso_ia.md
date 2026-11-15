# Registro de uso ético y responsable de IA

En este documento incluimos el registro de prompts y las herramientas empleadas.

## 1. Herramientas empleadas
| Herramienta | Modelo / versión | Miembro del equipo |
|---|---|---|
| ChatGPT | GPT‑5.5 Instant | @csm8-um |
| Gemini | 3.5 Flash-Lite  | @lmc08840 |

## 2. Prompts

### Registro de prompts de ChatGPT @csm8-um
Enlace público: 
https://chatgpt.com/share/6afa14bc-4164-83eb-a734-14d0708ee20a

**Prompt Límites clínicos de valores imposibles.**
- Instrucción: «Actúa como médico experto en medicina cardiovascular, que trabaja estrictamente en la evidencia científica. Qué valores máximos y mínimos […] son imposibles y sirven para detectar valores imposibles, según organismos o instituciones científicas médicas acreditadas…»
- Objetivo: fijar límites de plausibilidad con fuentes clínicas. El equipo contrastó la respuesta con esas fuentes y discutió casos concretos: IMC < 10, FC < 30 lpm, cintura < 40–50 cm, HbA1c y glucosa.

**Prompts Configuración del proyecto en GitHub.**
- Instrucción: «Que debe incluir el archivo .gitignore […] Explica las decisiones que tomas.»
Objetivo: preparar .gitignore, .gitattributes y la estructura de carpetas. Se rechazó excluir data/raw/, porque el repositorio quedaría sin datos y el cuaderno no se podría ejecutar.
Prompts – Detección de anomalías.
Instrucción: «Como podría descartar que se trata de errores de medicion y son valores artificiales que generan ruido?»
- Objetivo: analizar:
los valores negativos del alcohol;
los valores repetidos en cuatro variables, con su frecuencia, número de decimales y visualizaciones;
la ampliación del validador validar_datos_cardiovasculares() con valores extremos e incoherencias entre variables.

**Prompts Mecanismo de los valores ausentes.**
- Instrucción: «Quiero controlar si las ausencias dependen de la clase o de otras variables.»
- Objetivo: sustituir el pairplot por contrastes χ² por clase, la coincidencia de ausencias entre variables y el recuento de pacientes completos. El equipo simplificó el código propuesto.

**Prompts Relación de los predictores con el riesgo.**
- Instrucción: «Hay versiones no paramétricas de esas dos ideas?»
- Objetivo: entender y validar la correlación de Spearman, Kruskal-Wallis con ε², χ² con la V de Cramér y la matriz de correlaciones entre predictores. El equipo revisó cada interpretación antes de incorporarla.

**Prompts Desbalanceo y estrategia de evaluación.**
- Instrucción: «Pondría primero class_weight='balanced' y el remuestreo como alternativa que se compara. Que te parece?»
- Objetivo: definir la estratificación, el tratamiento del desbalanceo y f1_macro como métrica de optimización.

**Prompts Revisión de las decisiones de limpieza (apartados 4 y 5).**
- Instrucción: «Revisalo respecto a la revisión de las decisiones que has indicado anteriormente para validar la hipótesis de los 420 registros con valores repetidos.»
- Objetivo: revisar la redacción de los apartados 4.1 a 4.5, el código de limpieza.py y reglas_datos.py, y la comprobación del LDL con la fórmula de Friedewald. El equipo corrigió a ChatGPT: excluir los casos LDL > total introducía un sesgo de selección.

**Prompt Auditoría del cuaderno.**
- Instrucción: «Realiza un analisis detallado y exhaustivo del notebook adjunto hasta la linea “## 6”. Comprueba que los datos son correctos, completos y coherentes…»
- Objetivo: verificar la coherencia entre el texto y las salidas del cuaderno hasta el apartado 5.


### Gemini @lmc08840

**Prompt  – Diagnóstico inicial y hoja de ruta (con archivo adjunto .ipynb).**
- Objetivo. Establecer los pasos posteriores al Análisis Exploratorio de Datos (EDA) realizado previamente (con apoyo de ChatGPT) para preparar el dataset clínico de 2.254 registros y 28 variables antes del modelado.

**Prompts para código y representación gráfica  Análisis de los modelos simples que se puedan usar como clasificadores, análisis de modelos de ensambles y uso de la librería plot.**
- Objetivo. 
    - Ante la falta de tiempo para conciliar vida laboral, familiar y los estudios del máster, tenemos que reconocer que hemos usado la IA para la generación de código, ya que no tenemos una base en el uso de lenguaje Python y todavía menos en análisis estadístico. 
    - Hemos intentado ver los vídeos y seguir las clases, aunque desgraciadamente varias de ellas coinciden con nuestras propias horas de docencia en el CIFP Carlos III. Por ello, muchas de las cosas vistas en la tarea, las hemos ido aprendiendo a medida que trabajamos con ellas, por lo que igual que nos ocurre a diario en nuestro centro, la metodología seguida ha sido más de pedir ayuda a la IA y luego aprender cómo se resuelven los problemas.


**Prompt  Generación y estructuración de apartados de la memoria técnica:**
- Instrucción: "[Ej. Redacta la estructura y los apartados técnicos de la memoria del proyecto de predicción de riesgo cardiovascular, usando el notebook]"
- Objetivo: Organizar y redactar la documentación técnica del proyecto (metodología, tratamiento de datos, justificación de algoritmos y análisis de resultados) manteniendo rigor académico.

**Prompt  Generación de la memoria de uso ético de IA:**
- Instrucción: "Te voy a pasar un índice sobre un documento de uso ético y responsable de la IA. Quiero que lo rellenes con todo el contexto de esta conversación -> [Índice del apartado 4.3]"
- Objetivo: Documentar de forma transparente las herramientas utilizadas, el registro de instrucciones y la validación crítica de los resultados obtenidos.
