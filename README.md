# BPM Tri-metodológico: PLS-SEM + ANN + fsQCA

Implementación reproducible, íntegramente en código abierto, del análisis
tri-metodológico de un modelo jerárquico de **Business Process Management (BPM)**:
modelos de ecuaciones estructurales por mínimos cuadrados parciales (**PLS-SEM**),
redes neuronales artificiales (**ANN**) y análisis cualitativo comparativo
difuso (**fsQCA**).

> **English summary.** Fully open-source reproduction package for a
> tri-methodological BPM study combining PLS-SEM, artificial neural networks and
> fuzzy-set QCA. Every numerical result and figure of the article is regenerated
> by a single script, available in both Python and R, in Spanish and English
> versions. Total software licence cost: **$0**.

**Autor:** Henry Lizano-Mora
**Instituciones:** Instituto Tecnológico de Costa Rica / Universidad de Sevilla

---

## 1. Contenido del repositorio

| Archivo | Lenguaje | Idioma | Descripción |
|---|---|---|---|
| `reproducir_articulo.py` | Python ≥ 3.9 | Español | Script maestro de reproducción. Implementa PLS-SEM, ANN y fsQCA **desde cero** (sin dependencias PLS/QCA especializadas), en 7 módulos. Genera figuras en `figs/`. |
| `reproduce_article.py` | Python ≥ 3.9 | Inglés | Traducción literal del anterior. Misma lógica numérica y mismas semillas. Genera figuras en `figs_en/`. |
| `reproducir_articulo.R` | R ≥ 4.0 | Español | Réplica independiente usando los paquetes canónicos del campo (`seminr`, `QCA`, `nnet`). Sirve de **validación cruzada** de la implementación en Python. Figuras en `figs/`. |
| `reproduce_article_en.R` | R ≥ 4.0 | Inglés | Traducción del anterior. Figuras en `figs_en/`. |
| `requirements.txt` | — | — | Dependencias de Python con versiones fijadas (*pinned*), incluidas las transitivas, para reproducibilidad bit a bit. |
| `final_dataset_plssem.csv` | — | — | Conjunto de datos analítico: **n = 56** observaciones válidas, 20 indicadores en escala Likert. Sin identificadores personales. |
| `LICENSE` | — | — | Licencia MIT (véase §7). |
| `CITATION.cff` | — | — | Metadatos de citación legibles por máquina (GitHub, Zenodo, gestores bibliográficos). |

Las dos implementaciones (Python y R) son **independientes**: la de Python
programa los algoritmos explícitamente, mientras que la de R delega en paquetes
consolidados. La convergencia sustantiva de ambas es, en sí misma, evidencia de
robustez de los hallazgos; pueden aparecer diferencias marginales en la tercera
cifra decimal por diferencias de semillas y generadores pseudoaleatorios.

---

## 2. Ejecución

Todos los scripts esperan `final_dataset_plssem.csv` **en el mismo directorio**;
ejecútalos desde la raíz del repositorio.

### Python

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python reproducir_articulo.py      # o: python reproduce_article.py
```

### R

```r
install.packages(c("seminr", "QCA", "nnet", "NeuralNetTools",
                   "caret", "ggplot2", "reshape2"))
source("reproducir_articulo.R")    # o: source("reproduce_article_en.R")
```

Tiempo de ejecución aproximado en un portátil actual: **1–3 minutos** en Python
(dominado por el *bootstrap* de 3 000 remuestreos y las 330 redes neuronales) y
algo más en R.

---

## 3. Modelo de medida y datos

El modelo es un **modelo de constructos jerárquicos (HCM)** estimado en dos
etapas. Los 20 indicadores del CSV se agrupan así:

| Constructo | Rol | Indicadores |
|---|---|---|
| **SA** — Alineamiento estratégico | Reflectivo (dimensión de 1.er orden) | `Ae1`, `Ae2`, `Ae3` |
| **PEO** — Orientación a procesos | Reflectivo (dimensión de 1.er orden) | `Ep1`–`Ep5` |
| **GOVMEAS** — Gobernanza y medición | Reflectivo (dimensión de 1.er orden) | `Co1`–`Co5`, `Co8`, `Md1`, `Md3`, `Md4` |
| **BPM_Capability** | **Formativo** (constructo de 2.º orden) | Scores de SA, PEO y GOVMEAS |
| **BPO** — Desempeño de procesos | Compuesto (variable dependiente) | `Co7`, `Ae10`, `Md7` |

El *score* de cada dimensión reflectiva es la media de sus indicadores
estandarizados (*z*-scores); es el enfoque de **puntuaciones de constructo en dos
etapas** habitual en HCM.

---

## 4. Los tres algoritmos

Cada script se organiza en los mismos siete módulos, de modo que la
correspondencia Python ↔ R es línea a línea.

```
MÓDULO 0  Carga de datos y construcción de scores (Etapa 1)
MÓDULO 1  Modelo de medida: fiabilidad, AVE, HTMT
MÓDULO 2  PLS-SEM Etapa 2: modelo formativo + estructural + bootstrap
MÓDULO 3  Evaluación predictiva (Q² / PLSpredict)
MÓDULO 4  Redes neuronales artificiales (diagnóstico + robusto)
MÓDULO 5  fsQCA: calibración + necesidad + suficiencia con PRI
MÓDULO 6  Generación de figuras
```

### 4.1 PLS-SEM (Módulos 1–3)

**Qué responde:** ¿en qué medida la capacidad BPM explica y predice el desempeño
de procesos, y con qué peso contribuye cada dimensión? Es el análisis
**simétrico y lineal** del conjunto: estima un efecto promedio neto.

**Modelo de medida (Módulo 1).** Para cada constructo reflectivo se calculan:

- **α de Cronbach**, con la forma clásica
  `α = k/(k−1) · (1 − Σ var(xᵢ)/var(Σxᵢ))`.
- **Cargas externas** λᵢ como correlación de cada indicador con el *score* del
  constructo; umbral de referencia λ ≥ 0,708.
- **Fiabilidad compuesta (CR)** = `(Σλ)² / [(Σλ)² + Σ(1 − λᵢ²)]`.
- **Varianza media extraída (AVE)** = `media(λᵢ²)`; criterio AVE ≥ 0,50.
- **HTMT** (Henseler, Ringle & Sarstedt, 2015): cociente entre la media de las
  correlaciones *heterotrait–heteromethod* y la media geométrica de las
  correlaciones *monotrait–heteromethod* de cada bloque. Validez discriminante
  si HTMT < 0,90. En Python la matriz se calcula explícitamente
  (`htmt_matrix`); en R se obtiene de `seminr::estimate_pls`.

**Modelo estructural (Módulo 2).** El constructo de segundo orden
`BPM_Capability` se especifica como **compuesto formativo (Modo B)** sobre las
tres dimensiones estandarizadas, y `BPO` igualmente como compuesto sobre sus tres
indicadores. La función `pls_formative()` implementa el algoritmo PLS clásico con
**esquema de ponderación de camino** (*path weighting*):

1. Inicializar los pesos externos `wx`, `wy` uniformemente y normalizarlos.
2. **Aproximación externa:** calcular los *scores* latentes `X = X_ind·wx`,
   `Y = Y_ind·wy` y estandarizarlos.
3. **Aproximación interna:** ponderar cada constructo por el vecino de la red
   estructural usando el coeficiente de correlación `b` entre `X` e `Y`.
4. **Actualización de pesos (Modo B):** regresión OLS de cada constructo interno
   sobre sus propios indicadores (`np.linalg.lstsq`) — a diferencia del Modo A,
   que usaría correlaciones simples.
5. Reescalar los pesos para varianza unitaria del *score* y repetir desde (2)
   hasta que la suma de cambios absolutos sea < 10⁻⁷ (máx. 300 iteraciones).

Con los *scores* convergidos se obtienen el coeficiente de camino β (correlación
entre `X` e `Y`), **R²** = β² y **R² ajustado**. Adicionalmente se reportan:

- **Cargas formativas**: correlación de cada indicador con el *score* del
  constructo, para interpretar la contribución absoluta.
- **VIF** de los indicadores formativos (`vif_scores`), calculado como
  `1/(1 − R²ᵢ)` de la regresión de cada indicador sobre los demás; se busca
  VIF < 5 para descartar colinealidad, requisito específico de los modelos
  formativos.
- **Bootstrap no paramétrico** con **B = 3 000** remuestreos con reemplazo.
  En cada réplica se reestima el modelo completo y se corrige la **indeterminación
  de signo** típica de PLS (si β < 0 se invierten simultáneamente β y los pesos).
  De la distribución bootstrap se derivan errores estándar, estadísticos *t*,
  valores *p* e intervalos de confianza percentil al 95 %.

**Evaluación predictiva (Módulo 3).** El Python implementa **Q² de
Stone-Geisser** mediante *blindfolding* con distancia de omisión 7: se elimina
sistemáticamente cada séptima observación, se predice el indicador omitido a
partir del *score* del constructo y se acumula `Q² = 1 − SSE/SSO`; Q² > 0 indica
relevancia predictiva. El R usa `seminr::predict_pls` (**PLSpredict**, 10
particiones × 10 repeticiones), es decir, validación cruzada fuera de muestra
propiamente dicha. Es una diferencia deliberada entre implementaciones: la
versión R es el estándar metodológico, la versión Python es una aproximación
autocontenida sin dependencias externas.

### 4.2 Redes neuronales artificiales (Módulo 4)

**Qué responde:** ¿existen relaciones **no lineales** entre las dimensiones BPM y
el desempeño que PLS-SEM, por construcción lineal, no captaría? Las ANN se usan
aquí como complemento no lineal, no como sustituto.

Los tres *scores* (SA, PEO, GOVMEAS) y la variable objetivo BPO se normalizan a
`[0, 1]` con *min–max*. La importancia relativa de cada predictor se obtiene con
el **algoritmo de Garson (1991)**: se toman los valores absolutos de los pesos
entrada→oculta y oculta→salida, se multiplican por cada neurona oculta, se suman
por predictor y se expresan como porcentaje del total (`garson_importance`; en R,
`NeuralNetTools::garson`).

El módulo está deliberadamente dividido en dos partes, y esa división es un
resultado metodológico del artículo:

**(a) Diagnóstico — por qué una sola red no basta.** Se entrenan **30 redes
independientes** (`MLPRegressor`, 1 capa oculta de 10 neuronas, activación
`tanh`, `adam`, 2 000 iteraciones) que difieren **únicamente en la semilla de
inicialización**. El script tabula qué predictor resulta "dominante" en cada
semilla y el rango (máx − mín) de importancia por constructo. El resultado es que
la dominancia cambia con la semilla: **la ejecución única no es fiable, es un
artefacto de inicialización**. La figura 4 documenta esta inestabilidad.

**(b) Protocolo robusto.** Se entrenan **10 redes × validación cruzada de 10
particiones = 100 modelos**, con dos cambios clave respecto al diagnóstico:
capa oculta reducida a **3 neuronas** y **regularización L2** (`alpha = 0.1` en
scikit-learn, `decay = 0.1` en `nnet`). Las importancias se promedian dentro de
cada red y luego entre redes, reportando media y desviación típica. Se controla
el sobreajuste comparando **RMSE de entrenamiento frente a RMSE de prueba**: la
diferencia resulta prácticamente nula (≈ −0,006 en la ejecución de referencia),
lo que descarta el sobreajuste. Frente al diagnóstico de ejecución única, el
promediado sobre 100 modelos regularizados produce un **ordenamiento estable** de
las importancias, que es la magnitud que el artículo interpreta; la desviación
típica residual entre redes se reporta explícitamente junto a cada media y se
visualiza como barra de error en la figura 5.

### 4.3 fsQCA (Módulo 5)

**Qué responde:** ¿qué **combinaciones** de condiciones son suficientes para un
desempeño alto? A diferencia de PLS-SEM, fsQCA es **asimétrica y configuracional**:
admite equifinalidad (varios caminos al mismo resultado) y causalidad asimétrica
(las causas del desempeño alto no son la imagen especular de las del bajo).

**Calibración.** Los *scores* continuos se transforman en pertenencias difusas
`[0, 1]` mediante calibración directa por **percentiles**: umbral de pertenencia
plena `P95`, punto de cruce `P50` y umbral de exclusión plena `P05`. La función
`calibrate()` de Python aplica una interpolación lineal por tramos entre esos
tres anclajes, recortada a `[0.05, 0.95]` para evitar los valores extremos 0 y 1;
la versión R delega en `QCA::calibrate(type = "fuzzy")`, que usa la
transformación logística estándar. Ambas producen el mismo ordenamiento; la
versión Python es una aproximación lineal más simple.

**Análisis de necesidad.** Para cada condición se calcula

- **Consistencia** `= Σ min(condición, resultado) / Σ resultado`
- **Cobertura** `= Σ min(condición, resultado) / Σ condición`

Una condición se considera necesaria si su consistencia ≥ **0,90**. El análisis
se ejecuta **dos veces**: para el desempeño alto (BPO) y para su negación
(~BPO = 1 − BPO), precisamente para poner a prueba la asimetría causal.

**Análisis de suficiencia.** Se construye la **tabla de verdad** completa
recorriendo los 2³ = 8 rincones del espacio de propiedades. La pertenencia de
cada caso a un rincón es el mínimo (operador AND difuso) de sus condiciones,
negadas mediante `1 − x` donde corresponda. Para cada configuración se calcula:

- **Consistencia de suficiencia** `= Σ min(m, BPO) / Σ m`, umbral ≥ **0,80**.
- **PRI** (*Proportional Reduction in Inconsistency*), que penaliza las
  configuraciones simultáneamente consistentes con el resultado y su negación:
  `PRI = [Σ min(m,Y) − Σ min(min(m,Y), min(m,~Y))] / [Σ m − Σ min(min(m,Y), min(m,~Y))]`,
  umbral ≥ **0,70**. Es el filtro que descarta las soluciones espurias.
- **Cobertura bruta** de cada configuración retenida.

Las configuraciones que superan ambos umbrales se combinan con el operador OR
difuso (máximo) para obtener la **consistencia y cobertura de la solución
global**. El script informa además en cuántas de las configuraciones suficientes
aparece GOVMEAS, criterio con el que se identifica como **condición central**.
La versión R obtiene la solución mediante `QCA::truthTable` + `QCA::minimize`,
que aplica el algoritmo de minimización booleana de Quine-McCluskey.

### 4.4 Síntesis tri-metodológica (Módulo 6)

El gráfico radar final superpone los tres perfiles de importancia —pesos
formativos de PLS-SEM, importancia de Garson de las ANN y consistencia de
necesidad de fsQCA—, cada serie normalizada por su propio máximo para hacerlos
comparables en forma, no en escala. La convergencia de los tres métodos sobre la
misma dimensión es el argumento central del artículo.

---

## 5. Figuras generadas

Los scripts en español escriben en `figs/` y los ingleses en `figs_en/`
(ambos directorios se crean automáticamente y están excluidos del control de
versiones):

| Figura | Contenido | Python | R |
|---|---|:---:|:---:|
| `fig2_cargas` / `fig2_loadings` | Cargas externas por constructo, con umbral 0,708 | ✓ | ✓ |
| `fig3_htmt` | Matriz HTMT (mapa de calor triangular) | ✓ | — |
| `fig4_ann_inestable` / `fig4_ann_unstable` | Inestabilidad de la importancia en 30 redes de ejecución única | ✓ | — |
| `fig5_ann_robusto` / `fig5_ann_robust` | Importancia robusta (10 redes × 10 particiones) con barras de error | ✓ | ✓ |
| `fig7_fsqca_xy` | Diagrama XY de suficiencia de la configuración central | ✓ | ✓ |
| `fig8_radar` / `fig9_radar` | Radar de síntesis tri-metodológica | ✓ | ✓ |

---

## 6. Reproducibilidad

- Semilla global fija: `np.random.seed(42)` / `set.seed(42)`.
- Semillas deterministas por red: `0…29` en el diagnóstico y `100+net` en el
  protocolo robusto.
- Bootstrap con semilla fija (`seed = 42` en `seminr::bootstrap_model`).
- Dependencias de Python fijadas por versión exacta en `requirements.txt`,
  incluidas las transitivas.

Con ello, cada ejecución devuelve los mismos números. Las diferencias entre la
implementación en Python y la de R se limitan a la tercera cifra decimal y se
deben a generadores pseudoaleatorios y rutinas de optimización distintas.

---

## 7. Licencia

Este proyecto se distribuye bajo la **Licencia MIT** (véase [`LICENSE`](LICENSE)).

**Por qué MIT y no GPL o Apache-2.0.** El objetivo es la máxima difusión
científica: MIT es la licencia permisiva más corta y más reconocida, no impone
condiciones a quien reutilice el código (ni copyleft ni obligaciones de aviso
extensas), es aceptada sin fricción por repositorios académicos como Zenodo y
figShare y por las políticas de disponibilidad de código de las principales
editoriales, y es **compatible con la GPL**, de modo que otros investigadores
pueden incorporar estas rutinas tanto en proyectos GPL como en desarrollos
propietarios o comerciales. La GPL habría restringido esa segunda vía y la
Apache-2.0, aunque añade una concesión expresa de patentes, introduce requisitos
de aviso y es incompatible con la GPL-2, lo que complicaría la reutilización
junto a paquetes de R con esa licencia.

**Nota sobre las dependencias.** El código propio de este repositorio es MIT.
Los paquetes de R que invoca (`seminr`, `QCA`, `nnet`, `caret`, entre otros) se
distribuyen bajo GPL y conservan sus propias licencias; quien redistribuya una
obra combinada con ellos deberá respetar los términos de la GPL. Las
dependencias de Python (NumPy, pandas, SciPy, scikit-learn, Matplotlib, seaborn)
son todas permisivas (BSD/MIT).

---

## 8. Citación

Si utilizas este código o los datos, cita el artículo asociado y este
repositorio. El archivo [`CITATION.cff`](CITATION.cff) contiene los metadatos en
formato legible por máquina; GitHub muestra automáticamente el botón *Cite this
repository* a partir de él.

---

## 9. Referencias metodológicas

- Garson, G. D. (1991). Interpreting neural-network connection weights.
  *AI Expert*, 6(4), 46–51.
- Hair, J. F., Hult, G. T. M., Ringle, C. M., & Sarstedt, M. (2022).
  *A Primer on Partial Least Squares Structural Equation Modeling (PLS-SEM)*
  (3.ª ed.). Sage.
- Henseler, J., Ringle, C. M., & Sarstedt, M. (2015). A new criterion for
  assessing discriminant validity in variance-based structural equation
  modeling. *Journal of the Academy of Marketing Science*, 43(1), 115–135.
- Ragin, C. C. (2008). *Redesigning Social Inquiry: Fuzzy Sets and Beyond*.
  University of Chicago Press.
- Sarstedt, M., Hair, J. F., Cheah, J.-H., Becker, J.-M., & Ringle, C. M. (2019).
  How to specify, estimate, and validate higher-order constructs in PLS-SEM.
  *Australasian Marketing Journal*, 27(3), 197–211.
- Shmueli, G., Ray, S., Velasquez Estrada, J. M., & Chatla, S. B. (2016).
  The elephant in the room: Predictive performance of PLS models.
  *Journal of Business Research*, 69(10), 4552–4564.
