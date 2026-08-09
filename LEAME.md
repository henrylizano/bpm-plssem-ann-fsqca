# BPM Tri-metodológico: PLS-SEM + ANN + fsQCA

Implementación reproducible, íntegramente en código abierto, del análisis
tri-metodológico de un modelo jerárquico de **Business Process Management (BPM)**:
modelos de ecuaciones estructurales por mínimos cuadrados parciales (**PLS-SEM**),
redes neuronales artificiales (**ANN**) y análisis cualitativo comparativo
difuso (**fsQCA**).

> 🇬🇧 **English version:** [`README.md`](README.md)

**Autor:** Henry Lizano-Mora
**Instituciones:** Instituto Tecnológico de Costa Rica / Universidad de Sevilla

---

## 1. Contenido del repositorio

| Archivo | Lenguaje | Idioma | Descripción |
|---|---|---|---|
| `reproduce_article.py` | Python ≥ 3.9 | Inglés | **Implementación de referencia.** Programa PLS-SEM, ANN y fsQCA **desde cero** (sin dependencias PLS/QCA especializadas), en 7 módulos, replicando las configuraciones canónicas de R. Genera figuras en `figs_en/`. |
| `reproducir_articulo.py` | Python ≥ 3.9 | Español | Versión en español de la misma arquitectura de 7 módulos. Genera figuras en `figs/`. Los módulos 3 y 5 conservan las variantes simplificadas previas (Q² por *blindfolding* y calibración lineal por tramos); véase la nota al final de esta sección. |
| `reproducir_articulo.R` | R ≥ 4.0 | Español | Réplica independiente usando los paquetes canónicos del campo (`seminr`, `QCA`, `nnet`). Sirve de **validación cruzada** de la implementación en Python. Figuras en `figs/`. |
| `reproduce_article_en.R` | R ≥ 4.0 | Inglés | Traducción del anterior. Figuras en `figs_en/`. |
| `requirements.txt` | — | — | Dependencias de Python con versiones fijadas (*pinned*), incluidas las transitivas, para reproducibilidad bit a bit. |
| `final_dataset_plssem.csv` | — | — | Conjunto de datos analítico: **n = 56** observaciones válidas, 20 indicadores en escala Likert. Sin identificadores personales. |
| `references/` | — | — | Bibliografía del artículo (63 entradas) en tres formatos: **BibTeX**, **BibLaTeX** y **Zotero RDF**. Véase [`references/README.md`](references/README.md). |
| `LICENSE` | — | — | Licencia MIT (véase §7). |
| `CITATION.cff` | — | — | Metadatos de citación legibles por máquina (GitHub, Zenodo, gestores bibliográficos). |

Las dos implementaciones (Python y R) son **independientes**: la de Python
programa los algoritmos explícitamente, mientras que la de R delega en paquetes
consolidados. La convergencia sustantiva de ambas es, en sí misma, evidencia de
robustez de los hallazgos; pueden aparecer diferencias marginales en la tercera
cifra decimal por diferencias de semillas y generadores pseudoaleatorios.

> **Nota sobre `reproducir_articulo.py`.** `reproduce_article.py` incorpora tres
> mejoras metodológicas que la versión en español todavía no recoge:
> PLSpredict completo con contraste frente a un modelo lineal (módulo 3),
> optimizador L-BFGS con activación logística en las ANN (módulo 4) y
> calibración logística de Ragin con solución conservadora y análisis de
> sensibilidad (módulo 5). Para reproducir los resultados publicados, usa la
> versión en inglés.

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

Tiempo de ejecución aproximado en un portátil actual: **2–5 minutos** en Python
(dominado por el *bootstrap* de 3 000 remuestreos, las 100 particiones de
PLSpredict y las 130 redes neuronales) y algo más en R.

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
MÓDULO 3  Evaluación predictiva (PLSpredict: PLS frente a benchmark lineal)
MÓDULO 4  Redes neuronales artificiales (diagnóstico + robusto)
MÓDULO 5  fsQCA: calibración logística + necesidad/RoN + solución conservadora
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

**Evaluación predictiva (Módulo 3).** El módulo implementa **PLSpredict**
(Shmueli *et al.*, 2016, 2019) tal como se define metodológicamente: validación
cruzada de **10 particiones × 10 repeticiones** fuera de muestra. Dentro de cada
partición:

1. Se estandarizan los datos **con los estadísticos del subconjunto de
   entrenamiento** (nunca con los del conjunto completo), para evitar
   *data leakage*.
2. Se estima el modelo PLS solo con el entrenamiento, corrigiendo la
   indeterminación de signo, y se proyecta el *score* latente de las
   observaciones retenidas.
3. Se predicen los indicadores endógenos como `β · score · carga` y se acumulan
   los errores de predicción.
4. En paralelo se ajusta el **benchmark lineal (LM)**: una regresión OLS de cada
   indicador endógeno sobre los tres indicadores exógenos.

Se reportan **RMSE** y **MAE** de ambos modelos por indicador, más el
**Q²predict** (que usa la media de entrenamiento como referencia ingenua). El
criterio de Shmueli *et al.* (2019) clasifica la capacidad predictiva según en
cuántos indicadores el modelo PLS bate al LM: 3/3 alta, 2/3 media, ≤ 1/3 baja.
El script imprime ese veredicto automáticamente en lugar de asumirlo. La
implementación en R usa `seminr::predict_pls` con los mismos parámetros.

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
particiones = 100 modelos**, con tres cambios clave respecto al diagnóstico:

- Arquitectura parsimoniosa: capa oculta reducida a **3 neuronas**.
- **Regularización L2**: `alpha = 0.1` en scikit-learn, `decay = 0.1` en `nnet`.
- **Réplica exacta de la configuración de R**: optimizador **L-BFGS** y
  activación **logística** (sigmoide), que es lo que usa `nnet` internamente
  (BFGS + sigmoide), en lugar del `adam` + `tanh` del módulo diagnóstico. Esto
  garantiza la equivalencia numérica del hallazgo entre plataformas.

Las importancias se promedian dentro de cada red y luego entre redes,
reportando media y desviación típica. Se controla el sobreajuste comparando
**RMSE de entrenamiento frente a RMSE de prueba**: la diferencia resulta
prácticamente nula (+0,004 en la ejecución de referencia). Con esta
configuración la desviación típica entre redes cae a **≤ 0,2 puntos
porcentuales**, frente a los rangos de decenas de puntos del diagnóstico: las
importancias pasan a ser **estables y reportables**, y se visualizan con barras
de error en la figura 5.

### 4.3 fsQCA (Módulo 5)

**Qué responde:** ¿qué **combinaciones** de condiciones son suficientes para un
desempeño alto? A diferencia de PLS-SEM, fsQCA es **asimétrica y configuracional**:
admite equifinalidad (varios caminos al mismo resultado) y causalidad asimétrica
(las causas del desempeño alto no son la imagen especular de las del bajo).

**Calibración.** Los *scores* continuos se transforman en pertenencias difusas
`[0, 1]` mediante la **calibración directa logística de Ragin**, con anclajes por
percentiles: pertenencia plena `P95`, punto de cruce `P50` y exclusión plena
`P05`. La función `calibrate()` calcula los *log-odds* escalados de forma que en
los umbrales extremos valgan `±log(0,95/0,05)` y aplica la sigmoide
`1/(1+e^−logodds)`. Es **numéricamente equivalente** a
`QCA::calibrate(type = "fuzzy", logistic = TRUE)` de R, de modo que ambas
implementaciones producen las mismas pertenencias.

**Análisis de necesidad.** Para cada condición se calcula

- **Consistencia (inclN)** `= Σ min(condición, resultado) / Σ resultado`
- **Cobertura (covN)** `= Σ min(condición, resultado) / Σ condición`
- **RoN** (*Relevance of Necessity*, Schneider & Wagemann, 2012)
  `= Σ(1 − condición) / Σ(1 − min(condición, resultado))`

El RoN es esencial: una condición puede alcanzar una consistencia alta
simplemente por ser **trivialmente omnipresente** (casi todos los casos
pertenecen a ella), y el RoN detecta ese caso reportando un valor bajo. Sin él,
una condición irrelevante puede parecer necesaria. El análisis se ejecuta **dos
veces**, para el desempeño alto (BPO) y para su negación (~BPO = 1 − BPO), a fin
de poner a prueba la asimetría causal.

**Análisis de suficiencia.** Se construye la **tabla de verdad** completa
recorriendo los 2³ = 8 rincones del espacio de propiedades. La pertenencia de
cada caso a un rincón es el mínimo (operador AND difuso) de sus condiciones,
negadas mediante `1 − x` donde corresponda. Para cada configuración se calcula:

- **Consistencia de suficiencia** `= Σ min(m, BPO) / Σ m`, umbral ≥ **0,80**.
- **PRI** (*Proportional Reduction in Inconsistency*), que penaliza las
  configuraciones simultáneamente consistentes con el resultado y su negación:
  `PRI = [Σ min(m,Y) − Σ min(min(m,Y), min(m,~Y))] / [Σ m − Σ min(min(m,Y), min(m,~Y))]`,
  umbral ≥ **0,70**. Es el filtro que descarta las soluciones espurias.
- **Número de casos** con pertenencia > 0,5. Las filas con `n = 0` se marcan
  explícitamente como **restos lógicos** (`OUT = ?`): configuraciones sin
  evidencia empírica.

**Minimización conservadora.** La función `minimize_conservative()` aplica el
algoritmo de **Quine-McCluskey** únicamente sobre las filas **observadas** con
`OUT = 1`, combinando iterativamente términos que difieren en una sola condición
y sustituyendo esa condición por un *don't care*; después selecciona los
implicantes primos esenciales y completa la cobertura de forma voraz. Los restos
lógicos **no** se incorporan como supuestos simplificadores, lo que produce la
**solución conservadora (compleja)** — la más exigente de las tres soluciones de
fsQCA, porque no asume nada sobre configuraciones que no se observaron. Para
cada término se reportan consistencia, PRI, **cobertura bruta** y **cobertura
única** (la porción del resultado explicada solo por ese término); para la
solución global, consistencia, PRI y cobertura. En R el equivalente es
`QCA::truthTable` + `QCA::minimize`.

**Análisis de sensibilidad.** El módulo repite el procedimiento completo con
anclajes alternativos **90/50/10** en lugar de 95/50/5, y contrasta ambas
soluciones. Que la estructura de la solución y la centralidad de GOVMEAS se
mantengan bajo los dos esquemas es la evidencia de que el resultado no es un
artefacto de las decisiones de calibración.

### 4.4 Síntesis tri-metodológica (Módulo 6)

El gráfico radar final superpone los tres perfiles de importancia —pesos
formativos de PLS-SEM, importancia de Garson de las ANN y consistencia de
necesidad de fsQCA—, cada serie normalizada por su propio máximo para hacerlos
comparables en forma, no en escala. La convergencia de los tres métodos sobre la
misma dimensión es el argumento central del artículo.

Al terminar, el script imprime un bloque de resumen con los indicadores clave de
los tres métodos: β, R² y t de PLS-SEM; el recuento PLS-frente-a-LM de
PLSpredict; la importancia media y la desviación típica máxima entre redes de la
ANN; y la expresión de la solución conservadora de fsQCA con su consistencia,
PRI y cobertura. Todos los valores se interpolan de la ejecución en curso, no
están fijados en el código.

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
- Schneider, C. Q., & Wagemann, C. (2012). *Set-Theoretic Methods for the Social
  Sciences: A Guide to Qualitative Comparative Analysis*. Cambridge University
  Press.
- Sarstedt, M., Hair, J. F., Cheah, J.-H., Becker, J.-M., & Ringle, C. M. (2019).
  How to specify, estimate, and validate higher-order constructs in PLS-SEM.
  *Australasian Marketing Journal*, 27(3), 197–211.
- Shmueli, G., Ray, S., Velasquez Estrada, J. M., & Chatla, S. B. (2016).
  The elephant in the room: Predictive performance of PLS models.
  *Journal of Business Research*, 69(10), 4552–4564.
- Shmueli, G., Sarstedt, M., Hair, J. F., Cheah, J.-H., Ting, H., Vaithilingam, S.,
  & Ringle, C. M. (2019). Predictive model assessment in PLS-SEM: Guidelines for
  using PLSpredict. *European Journal of Marketing*, 53(11), 2322–2347.
