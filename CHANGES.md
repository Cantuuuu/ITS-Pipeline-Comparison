# Cambios introducidos en la rama `gabocorretions`

Documentación generada automáticamente — rama `gabocorretions` vs `master`.

---

## Resumen ejecutivo

Se agregaron dos módulos de análisis estadístico, se modificó `paper/main.tex` respondiendo a revisión, y se generaron siete archivos CSV de resultados.

| Tipo | Archivo | Estado |
|------|---------|--------|
| Paper | `paper/main.tex` | Modificado (+55 / −13 líneas) |
| Código | `forest_its/evaluation/statistical_tests.py` | Nuevo |
| Código | `forest_its/evaluation/site_sensitivity.py` | Nuevo |
| Resultado | `results/descriptive_stats_test.csv` | Nuevo |
| Resultado | `results/paired_tests.csv` | Nuevo |
| Resultado | `results/wilcoxon_tests.csv` | Nuevo (legacy) |
| Resultado | `results/bootstrap_ci.csv` | Nuevo |
| Resultado | `results/leave_one_site_out.csv` | Nuevo |
| Resultado | `results/ranking_stability.csv` | Nuevo |
| Resultado | `results/nibio_impact.csv` | Nuevo |

---

## 1. `forest_its/evaluation/statistical_tests.py`

### Propósito

Motor estadístico del proyecto. Toma `results/comparison_table_test.csv` y produce tres archivos de salida con estadísticas descriptivas, pruebas de hipótesis pareadas y bootstraps BCa. Diseñado para el contexto de n=11 parcelas donde los supuestos de normalidad no pueden verificarse.

### Constantes

**`METRICS`** — `["f1", "precision", "recall", "coverage"]`
Las cuatro métricas de evaluación del pipeline ITS.

**`COMPARISONS`** — cinco pares predefinidos:

| Etiqueta | Método A | Método B | Pregunta |
|----------|----------|----------|----------|
| `B_vs_D` | `rf` | `rf_density` | Efecto del density seeding en RF |
| `C_vs_E` | `pointnet2` | `pointnet2_density` | Efecto del density seeding en PN++ |
| `A_vs_D` | `baseline` | `rf_density` | Mejor pipeline vs baseline |
| `B_vs_C` | `rf` | `pointnet2` | RF vs PN++ con CHM |
| `D_vs_E` | `rf_density` | `pointnet2_density` | RF vs PN++ con Density |

La diferencia siempre se calcula como `B - A`; positivo = B gana.

### Funciones

#### `load_metrics(csv_path) -> pd.DataFrame`
Lee el CSV y valida columnas. Lanza `ValueError` explícito si falta alguna columna requerida. Retorna copia con exactamente siete columnas: `method, plot, institution, f1, precision, recall, coverage`.

#### `descriptive_stats(df, output_dir) -> pd.DataFrame`
Por cada combinación (método × métrica) calcula `mean`, `std` (ddof=1), `median`, `min`, `max`, `n`. Usa `ddof=1` (estimador insesgado) porque los datos son una muestra. Exporta a `descriptive_stats_test.csv` (20 filas: 5 métodos × 4 métricas).

#### `exact_permutation_test(x, y, alternative='two-sided') -> dict` — **PRUEBA PRINCIPAL**

Implementa el test de permutación exacto sobre diferencias pareadas `d = y - x`.

**Algoritmo:**
1. Calcula `observed = mean(d)`.
2. Enumera todas las `2^n` asignaciones de signo via `itertools.product([-1, 1], repeat=n)`. Con n=11 → 2048 permutaciones exactas.
3. Para cada asignación `s`, calcula `mean(s * d)`.
4. `p_value = #{|perm_means| >= |observed|} / 2^n` (bilateral).

**Por qué exacto sobre Monte Carlo:** con n=11 la enumeración completa toma fracciones de segundo y produce el p-valor correcto sin varianza de estimación — crítico cuando los p-valores caen cerca de umbrales de decisión.

Retorna: `{observed_mean_diff, p_value, n_permutations, n_pairs}`.

#### `_wilcoxon_legacy(a_vals, b_vals, n_nonzero) -> tuple` — privada
Encapsula `scipy.stats.wilcoxon(method='exact')` con manejo de errores. Requiere `n_nonzero >= 2`; si no, retorna `(nan, nan, "insuficientes_pares_no_nulos")`. Usada como referencia secundaria.

#### `paired_tests(df, output_dir) -> pd.DataFrame` — reemplaza `wilcoxon_tests`
Para cada (comparación, métrica) ejecuta **ambos** tests y registra todos los estadísticos lado a lado. Opera sobre `f1`, `precision`, `recall` (excluye `coverage`). Alinea los métodos por `plot` via `index.intersection` para garantizar comparación correcta. Exporta a `paired_tests.csv` (15 filas: 5 comparaciones × 3 métricas).

#### `wilcoxon_tests(df, output_dir) -> pd.DataFrame` — **LEGACY**
Versión anterior, conservada para compatibilidad. Ejecuta solo Wilcoxon. Exporta a `wilcoxon_tests.csv`. Marcada con `[LEGACY]` en docstring. No es llamada por `main()`.

#### `bootstrap_bca_ci(data, n_boot=10000, ci=0.95, seed=42) -> tuple` — **reemplaza percentil simple**

Calcula IC bootstrap BCa (Bias-Corrected and Accelerated) para la media. Corrige dos fuentes de error del percentil simple:

**Corrección de sesgo (z0):**
```
prop_less = #{boot_means < observed} / n_boot
z0 = Φ⁻¹(prop_less)   [con clip en (1e-6, 1-1e-6) para evitar ±∞]
```

**Aceleración (a_hat) via jackknife:**
```
jack_means[i] = mean(data sin elemento i)
a_hat = Σ(jack_mean - jack_means)³ / [6 · (Σ(jack_mean - jack_means)²)^1.5]
```
Si `denom == 0`, `a_hat = 0` (BCa degenera a BC).

**Ajuste de percentiles (fórmula de Efron & Tibshirani 1993, cap. 14):**
```
alpha1 = Φ(z0 + (z0 + z_α)  / (1 - a_hat·(z0 + z_α)))
alpha2 = Φ(z0 + (z0 + z_1-α) / (1 - a_hat·(z0 + z_1-α)))
ci_lower = percentile(boot_means, 100·alpha1)
ci_upper = percentile(boot_means, 100·alpha2)
```

Retorna: `(ci_lower, ci_upper, boot_means_array)`.

#### `bootstrap_ci(df, output_dir, n_boot=10000, ci=0.95, seed=42) -> pd.DataFrame`
Orquesta el BCa para dos tipos de cantidades:
1. Media de cada métrica por método (20 registros).
2. Media de diferencias pareadas de F1 para cada comparación (5 registros).

Usa semillas derivadas determinísticamente (`range(seed, seed+10_000)`) — una semilla única por llamada a `bootstrap_bca_ci` garantizando reproducibilidad sin correlación entre ICs. Agrega columna `ci_method = "BCa"`. Exporta a `bootstrap_ci.csv`.

#### `_sig(p) -> str` — privada auxiliar
Convierte p-valor a símbolo: `***` (<0.01), `**` (<0.05), `*` (<0.10), `ns` (≥0.10), `N/A` (nan).

#### `main()`
Argumentos CLI: `--input` (default: `results/comparison_table_test.csv`), `--output-dir` (default: `results/`).

Orden de ejecución:
1. `load_metrics` → valida CSV
2. `descriptive_stats` → exporta + imprime tabla F1
3. `paired_tests` → exporta + imprime tabla con ambos p-values lado a lado
4. `bootstrap_ci` → exporta + imprime ICs BCa con flag `✓ CI excluye 0`
5. Bloque de conclusiones: para cada comparación indica si es significativa con el test de permutación

### Decisiones de diseño

| Decisión | Razón |
|----------|-------|
| Permutación exacta como prueba principal | n=11 permite enumeración completa; sin varianza de estimación |
| BCa sobre percentil simple | Corrige sesgo y heterocedasticidad, importante con métricas acotadas en [0,1] |
| Wilcoxon mantenido como legacy | Compatibilidad con versiones previas; permite verificación cruzada con revisores |
| Coverage excluido de pruebas de hipótesis | Métrica de cobertura geométrica, no de detección árbol-a-árbol; semánticamente distinta |
| Alineación por plot en comparaciones | Garantiza que se comparan los mismos sitios entre métodos |

---

## 2. `forest_its/evaluation/site_sensitivity.py`

### Propósito

Análisis LOSO (Leave-One-Site-Out) para demostrar que las conclusiones del paper no son un artefacto del sesgo de NIBIO, que aporta el 55% del test set (6 de 11 plots). Genera tres archivos CSV de robustez.

### Constantes

**`METHOD_ORDER`** — `["baseline", "rf", "pointnet2", "rf_density", "pointnet2_density"]`
Orden canónico de presentación en todas las tablas.

**`METHOD_LABELS`** — diccionario de traducción clave→etiqueta del paper (A/B/C/D/E).

**`MIN_PLOTS_WARNING = 5`** — umbral bajo el cual se emite `UserWarning` por inestabilidad estadística.

### Funciones privadas

#### `_load(csv_path) -> pd.DataFrame`
Idéntica en comportamiento a `load_metrics` de `statistical_tests.py`. Valida las siete columnas requeridas y retorna copia.

#### `_method_means(sub) -> dict[str, float]`
Helper: `sub.groupby("method")["f1"].mean().to_dict()`.

#### `_rank(means) -> list[str]`
Ordena métodos de mayor a menor `mean_f1`. Usa valor por defecto `-1` para métodos sin datos (caen al final).

### Funciones principales

#### `leave_one_site_out(df, output_dir) -> pd.DataFrame`

Para cada institución excluye todos sus plots y recalcula mean y std de F1 por método. Incluye condición `"none"` (conjunto completo) como referencia.

**Algoritmo:**
1. Versión full (`excluded_site = "none"`): agrupa por método, calcula stats, registra.
2. Bucle LOSO: para cada `site` filtra `df[df["institution"] != site]`, cuenta `n_remaining`, emite `UserWarning` si `n_remaining < 5`, calcula stats por método y registra.

Exporta a `leave_one_site_out.csv` (30 filas: 6 condiciones × 5 métodos).

#### `ranking_stability(loso_df, output_dir) -> pd.DataFrame`

Determina si el ranking de métodos cambia al excluir cada sitio.

**Detección de cambio:** compara únicamente los métodos comunes entre el ranking full y el ranking LOSO (evita falsos positivos por métodos ausentes). Si el orden de los métodos comunes difiere → `ranking_changed = True`.

Exporta a `ranking_stability.csv` (6 filas: una por condición de exclusión).

#### `nibio_impact(df, output_dir) -> pd.DataFrame`

Cuantifica exactamente cuánto cambia la media de F1 de cada método al incluir/excluir NIBIO.

- `delta = f1_without - f1_with` (positivo = NIBIO deprime las métricas)
- `pct_change = 100 * delta / f1_with`

Exporta a `nibio_impact.csv` (5 filas: una por método en orden canónico).

#### `_check_conclusions(df) -> list[tuple[str, bool, str]]`

Verifica las tres conclusiones centrales del paper **con y sin NIBIO**:

1. `rf_density` es el mejor flujo (`best == "rf_density"`).
2. Density seeding supera a CHM (`rf_density > rf` **y** `pn++_density > pn++`).
3. RF supera a PointNet++ (`rf > pointnet2` **y** `rf_density > pn++_density`).

Retorna 6 tuplas `(descripción, bool, detalle_numérico)` — 3 conclusiones × 2 escenarios.

#### `main()`
Argumentos CLI: `--input`, `--output-dir`.

Orden de ejecución:
1. `_load` → imprime distribución de plots por sitio y % de NIBIO
2. `leave_one_site_out` → imprime tabla pivot (excluded_site × method)
3. `ranking_stability` → imprime ranking con flag `✓ estable` / `⚠ CAMBIÓ`
4. `nibio_impact` → imprime delta con flechas de dirección
5. `_check_conclusions` → imprime estado de las 3 conclusiones (con y sin NIBIO)
6. Resumen final: una línea por escenario de exclusión con veredicto global

### Por qué es relevante el análisis LOSO

NIBIO representa el 55% del test y corresponde al bosque boreal noruego — el tipo de bosque más denso y difícil del benchmark. Si los métodos evaluados tuvieran rendimientos diferenciados en bosques boreales vs. otros biomas, las métricas agregadas reflejarían principalmente NIBIO. El LOSO es una **validación cruzada a nivel de dominio**: si las conclusiones sobreviven a la eliminación de cada sitio (especialmente NIBIO), son generalizables y no son un artefacto de composición del dataset.

---

## 3. Archivos CSV generados

### Árbol de dependencias

```
results/comparison_table_test.csv  ← fuente primaria (preexistente)
    │
    ├── statistical_tests.py
    │       ├── descriptive_stats_test.csv
    │       ├── paired_tests.csv          ← PRINCIPAL
    │       ├── wilcoxon_tests.csv        ← legacy
    │       └── bootstrap_ci.csv
    │
    └── site_sensitivity.py
            ├── leave_one_site_out.csv
            ├── ranking_stability.csv
            └── nibio_impact.csv
```

### `descriptive_stats_test.csv`
**20 filas** (5 métodos × 4 métricas). Columnas: `method, metric, mean, std, median, min, max, n`.

Responde: *¿Cuál es el rendimiento promedio de cada pipeline?*

### `paired_tests.csv`
**15 filas** (5 comparaciones × 3 métricas: f1, precision, recall). Columnas principales:

| Columna | Descripción |
|---------|-------------|
| `perm_p_value` | p-valor permutación exacta (2048 permutaciones) — **prueba principal** |
| `p_wilcoxon` | p-valor Wilcoxon exact — referencia secundaria |
| `mean_diff` | `B - A`; positivo = B gana |
| `n_nonzero_pairs` | pares con diferencia ≠ 0 (relevante para Wilcoxon) |
| `note` | error de Wilcoxon si falla; vacío normalmente |

Responde: *¿Son estadísticamente significativas las diferencias entre pipelines?*

**Resultados clave (F1):**

| Comparación | p_perm | p_wilcoxon | Δmean | Veredicto |
|-------------|--------|-----------|-------|-----------|
| B vs D: RF vs RF+Density | 0.0957 | 0.1230 | +0.067 | Marginal (*) |
| C vs E: PN++ vs PN++Density | 0.0664 | 0.1055 | +0.061 | Marginal (*) |
| A vs D: Baseline vs RF+Density | **0.0156** | 0.0195 | +0.097 | Significativa (**) |
| B vs C: RF vs PN++ | **0.0156** | 0.0195 | −0.060 | Significativa (**) |
| D vs E: RF+Density vs PN++Density | **0.0195** | 0.0195 | −0.066 | Significativa (**) |

### `wilcoxon_tests.csv`
**15 filas** — subconjunto legacy de `paired_tests.csv` sin las columnas de permutación. El campo de p-valor se llama `p_value` en lugar de `p_wilcoxon`.

### `bootstrap_ci.csv`
**25 filas** (20 de medias por método + 5 de diferencias pareadas F1). Columna `ci_method = "BCa"` siempre. Columnas principales: `type, label, metric, observed, ci_lower, ci_upper, n, n_boot, ci_level, ci_method`.

**Resultados clave (diferencias pareadas F1):**

| Comparación | Δ observado | BCa 95% CI | CI excluye 0 |
|-------------|------------|-----------|--------------|
| B vs D | +0.067 | [−0.020, +0.120] | No |
| C vs E | +0.061 | [−0.004, +0.105] | No |
| A vs D | **+0.097** | **[+0.042, +0.158]** | **Sí** |
| B vs C | **−0.060** | **[−0.125, −0.024]** | **Sí** |
| D vs E | **−0.066** | **[−0.108, −0.023]** | **Sí** |

Responde: *¿Con qué certeza podemos afirmar que una mejora es real?*

### `leave_one_site_out.csv`
**30 filas** (6 condiciones × 5 métodos). Columnas: `excluded_site, n_plots_remaining, method, mean_f1, std_f1`.

Responde: *¿Cómo cambia el rendimiento de cada método al excluir cada sitio?*

### `ranking_stability.csv`
**6 filas** (una por condición de exclusión). Columnas: `excluded_site, n_plots, rank_1..rank_5, ranking_changed`.

**Patrón clave:** `rf_density` ocupa `rank_1` en las 6 condiciones. `pointnet2` ocupa `rank_5` en las 6 condiciones. El orden de los métodos intermedios fluctúa.

Responde: *¿Es estable el ranking del mejor método independientemente del sitio excluido?*

### `nibio_impact.csv`
**5 filas** (una por método). Columnas: `method, f1_with_nibio, f1_without_nibio, delta, pct_change, n_plots_with, n_plots_without`.

**Resultados:**

| Método | F1 con NIBIO | F1 sin NIBIO | Δ | % cambio |
|--------|-------------|-------------|---|---------|
| baseline | 0.1119 | 0.1804 | +0.069 | +61.3% |
| rf | 0.1419 | 0.2043 | +0.062 | +44.0% |
| pointnet2 | 0.0823 | 0.1349 | +0.053 | +63.8% |
| rf_density | 0.2085 | 0.2772 | +0.069 | +32.9% |
| pointnet2_density | 0.1429 | 0.2005 | +0.058 | +40.3% |

Todos los deltas son positivos: NIBIO deprime sistemáticamente las métricas. `rf_density` es el más robusto (menor % de cambio relativo).

Responde: *¿Introduce NIBIO un sesgo cuantificable y en qué magnitud?*

---

## 4. Cambios en `paper/main.tex`

### Contexto

Commit `a1bdbba` — responde dos puntos de revisión:
- **Punto 3:** El test set es pequeño (11 plots, 323 árboles) y NIBIO pesa mucho (≈50%). El revisor pide reconocer esto y mostrar variabilidad.
- **Punto 4:** Las mejoras porcentuales (+47%, +74%) se reportaban sin pruebas formales. El revisor pide p-values e intervalos.

### Cambio 1 — Tablas 3 y 4: valores con media ± SD

Todos los valores de métricas de instancia ahora muestran `media ± desviación estándar` en lugar de solo la media. Afecta la Tabla 3 (resultados agregados) y la Tabla 4 (análisis factorial).

**Antes:**
```latex
& 0.138 & 0.133 & 0.112 & 0.133 \\
& \textbf{0.209} &
```

**Después:**
```latex
& $0.138 \pm 0.149$ & $0.133 \pm 0.143$ & $0.112 \pm 0.116$ & $0.133 \pm 0.143$ \\
& $\mathbf{0.209 \pm 0.161}$ &
```

El caption de la Tabla 3 también se actualizó: `"promedio sobre 11 plots"` → `"media $\pm$ SD sobre $n=11$ plots"`.

El párrafo posterior a la tabla agrega: la alta dispersión entre plots (SD comparable a la media) refleja la heterogeneidad estructural del benchmark, y el density seeding reduce la variabilidad de RF (SD: 0.226 → 0.161).

### Cambio 2 — Párrafo nuevo: análisis de significancia estadística

Se añade un párrafo `\paragraph{Análisis de significancia estadística.}` en la sección de análisis factorial (§6.4) que explica la elección del test de permutación exacto sobre Wilcoxon y t-Student, con justificación metodológica detallada.

### Cambio 3 — Tabla 5 nueva: significancia estadística

Nueva tabla `\label{tab:significance}` con columnas: Comparación, Δ̄ F1, SD(Δ), p_perm, IC 95% BCa (Δ), Plots con mejora.

| Comparación | Δ̄ F1 | p_perm | IC BCa | Mejora |
|-------------|-------|--------|--------|--------|
| RF: CHM→Density | +0.067 | 0.096* | [−0.004, +0.129] | 8/11 |
| PN++: CHM→Density | +0.061 | 0.066* | [+0.003, +0.110] | 9/11 |
| Baseline→RF+Density | +0.097 | 0.016** | [+0.027, +0.161] | 9/11 |
| RF vs PN++ (CHM) | −0.060 | 0.016** | — | — |
| RF vs PN++ (Density) | −0.066 | 0.020** | — | — |

`**` p<0.05, `*` p<0.10.

### Cambio 4 — §7.3 Limitaciones: tres sub-secciones

El párrafo genérico de limitaciones se reemplaza por tres `\paragraph{}` diferenciados:

**`\paragraph{Tamaño y balance del conjunto de evaluación.}`**
Cuantifica explícitamente la concentración de NIBIO: 6/11 plots (~50%) y 161/323 árboles (~50%). Reporta el IC BCa 95% de F1 para el mejor flujo: [0.128, 0.314]. Menciona que sitios individuales como CULS (F1=0.629) y NIBIO_5 (F1=0.000) dominan la dispersión.

**`\paragraph{Poder estadístico.}`**
Reconoce explícitamente que con n=11 el test de permutación exacto solo alcanza significancia marginal para el density seeding (p=0.096 RF, p=0.066 PN++). Señala que los intervalos BCa para PN++ excluyen el cero y que la mejora es consistente en 8–9/11 plots. Indica que FOR-instanceV2 es necesario para confirmar significancia estricta. Las comparaciones entre preprocesadores sí alcanzan significancia plena (p=0.016).

**`\paragraph{Otras limitaciones.}`**
Conserva el contenido anterior: PointNet++ con única configuración, conclusiones específicas al Watershed 3D volumétrico, especificidad a datos ULS.

### Cambio 5 — §8 Conclusiones: p-values integrados

La afirmación `"supera a PointNet++ MSG"` se actualiza a `"supera significativamente a PointNet++ MSG ($p=0.016$, test de permutación exacto)"`.

La afirmación sobre density seeding se actualiza a `"con significancia marginal ($p=0.096$ y $p=0.066$, respectivamente) limitada por el tamaño muestral del benchmark ($n=11$ plots)"`.

### Qué NO cambió

- Ninguna figura
- Ningún resultado numérico (los valores son los mismos; solo se agregó SD)
- Ninguna sección fue eliminada

---

## 5. Implicaciones para el paper

Los análisis confirman y matizan las conclusiones del paper:

**Confirmadas con significancia estadística:**
- RF+Density supera al baseline: p=0.016, BCa CI excluye 0.
- RF supera a PointNet++ (CHM y Density): p=0.016–0.020.
- El ranking `rf_density > ... > pointnet2` es estable en todos los escenarios LOSO.

**Matizadas:**
- La mejora del density seeding (B→D y C→E) es **marginal** estadísticamente (p=0.066–0.096) con n=11 plots, aunque el BCa CI casi excluye el 0 en C→E. La afirmación de "+47%/+74%" en F1 relativo es correcta pero debe reportarse como tendencia, no como resultado significativo a α=0.05.
- NIBIO deprime las métricas entre 33% y 64% según el método — las métricas reportadas en el paper son conservadoras.
