# Guía de Métricas — VAE PoreGen

Referencia de todas las métricas monitorizadas durante entrenamiento, validación y evaluación offline. Cada entrada indica qué mide la métrica, su fórmula, qué rango es sano y por qué existe.

**Formato de la tabla de monitorización:**

| Estado | Rango / señal | Acción |
|--------|--------------|--------|
| Sano ✓ | rango esperado | — |
| Aviso ⚠ | señal de alerta | qué revisar |
| Problema ✗ | señal de fallo | cómo intervenir |

---

## Índice

1. [Pérdidas](#1-pérdidas)
2. [Entrenamiento — train/](#2-entrenamiento--train)
3. [Validación — val/](#3-validación--val)
4. [Test — test/](#4-test--test)
5. [Monte Carlo — montecarlo/](#5-monte-carlo--montecarlo)
6. [Evaluación final completa — val\_full / test\_full](#6-evaluación-final-completa)
7. [Evaluación offline — eval\_checkpoint.py](#7-evaluación-offline)
8. [Imágenes TensorBoard](#8-imágenes-tensorboard)
9. [Archivos en disco](#9-archivos-en-disco)
10. [Tabla de prioridades](#10-tabla-de-prioridades)

---

## 1. Pérdidas

Calculadas en cada step de train y en cada paso de eval. La función principal es `compute_total_loss` en `losses/total.py`.

---

### `total` — Pérdida total del VAE

**Qué mide:** El único término que llama a `.backward()`. Suma los tres objetivos del modelo: reconstrucción XCT, segmentación de poros y regularización del espacio latente.

**Fórmula:**

$$\mathcal{L}_\text{total} = w_\text{xct} \cdot \mathcal{L}_\text{xct} + \mathcal{L}_\text{mask} + \beta \cdot \mathcal{L}_\text{KL}$$

| Estado | Rango / señal | Acción |
|--------|--------------|--------|
| Sano ✓ | Descenso monótono y suave | — |
| Aviso ⚠ | val sube mientras train baja | Posible overfitting; revisar regularización |
| Problema ✗ | Oscilaciones grandes o divergencia | Reducir lr o aumentar batch_size |

**Por qué está aquí:** Es el único escalar que el optimizador minimiza; si no baja, nada funciona.

---

### `xct_loss` — Reconstrucción XCT (Charbonnier)

**Qué mide:** Diferencia entre el volumen XCT reconstruido y el ground truth, en espacio z-score (sin activación en la salida del decoder).

**Fórmula:**

$$\mathcal{L}_\text{xct} = \frac{1}{N}\sum_{i} \sqrt{(\hat{x}_i - x_i)^2 + \varepsilon^2}, \quad \varepsilon = 10^{-6}$$

| Estado | Rango / señal | Acción |
|--------|--------------|--------|
| Sano ✓ | < 0.02 | — |
| Aviso ⚠ | Estancado en 0.03–0.05 | El decoder no recupera textura; revisar capacidad o lr |
| Problema ✗ | No baja de 0.05 tras muchos steps | Posible colapso latente; revisar KL |

**Por qué está aquí:** Charbonnier es diferenciable en 0 (a diferencia de L1) y menos sensible a outliers que MSE. Opera en z-score porque el decoder no tiene activación de salida.

---

### `mask_bce` — Binary Cross-Entropy con pos\_weight (o Focal Loss)

**Qué mide:** Error de clasificación vóxel a vóxel entre la máscara predicha y el ground truth, compensando el desequilibrio de clases (~5.5% poros).

**Fórmula BCE:**

$$\mathcal{L}_\text{BCE} = -\frac{1}{N}\sum_i \bigl[w^+ \cdot y_i \log\sigma(\hat{y}_i) + (1-y_i)\log(1-\sigma(\hat{y}_i))\bigr], \quad w^+ = 17.16$$

**Fórmula Focal (R04+):**

$$\mathcal{L}_\text{Focal} = -\frac{1}{N}\sum_i \alpha_t (1-p_t)^\gamma \log p_t, \quad \gamma=2,\; \alpha=0.25$$

| Estado | Rango / señal | Acción |
|--------|--------------|--------|
| Sano ✓ | Descenso continuo | — |
| Aviso ⚠ | Se estanca pronto sin bajar más | La red aprende la frecuencia base pero no la forma; revisar Tversky |
| Problema ✗ | Sube tras bajar | Colapso o lr demasiado alto |

**Por qué está aquí:** Sin pos\_weight, predecir siempre «no poro» da ~94.5% de accuracy trivialmente. Con $w^+=17.16$, un falso negativo cuesta 17× más que un falso positivo.

---

### `mask_tversky` (o `mask_dice`) — Pérdida por región

**Qué mide:** Desacuerdo de solapamiento geométrico entre la máscara predicha y el GT, con sesgo hacia minimizar falsos negativos.

**Fórmula Tversky:**

$$\mathcal{L}_\text{Tversky} = 1 - \frac{\text{TP} + 1}{\text{TP} + \alpha\,\text{FP} + \beta\,\text{FN} + 1}, \quad \alpha=0.3,\; \beta=0.7$$

**Fórmula Dice** ($\alpha=\beta=0.5$):

$$\mathcal{L}_\text{Dice} = 1 - \frac{2\,\text{TP} + 1}{2\,\text{TP} + \text{FP} + \text{FN} + 1}$$

| Estado | Rango / señal | Acción |
|--------|--------------|--------|
| Sano ✓ | 0.0–0.3, bajando | — |
| Aviso ⚠ | Baja más despacio que mask_bce | La red acierta la frecuencia global pero no la forma |
| Problema ✗ | Oscila o sube | Gradientes ruidosos; reducir mask_dice_weight |

**Por qué está aquí:** BCE da gradiente vóxel a vóxel pero no a nivel de objeto. Tversky/Dice complementa con gradiente de solapamiento global. Con $\beta=0.7 > \alpha=0.3$ se penalizan más los FN, priorizando recall para poros pequeños y dispersos.

---

### `mask_total` — Pérdida combinada de máscara

**Qué mide:** Suma ponderada de BCE (o Focal) + Tversky (o Dice).

**Fórmula:**

$$\mathcal{L}_\text{mask} = w_\text{bce} \cdot \mathcal{L}_\text{BCE} + w_\text{dice} \cdot \mathcal{L}_\text{Tversky}$$

**Por qué está aquí:** BCE sola converge a predecir la frecuencia base sin forma. Tversky sola tiene gradientes ruidosos al inicio. La combinación da señal complementaria y convergencia más estable.

---

### `kl` — Divergencia KL

**Qué mide:** Distancia entre la distribución posterior aprendida $q(z|x)$ y la prior $p(z) = \mathcal{N}(0,I)$. Con free-bits, los canales por debajo del umbral se clampean para evitar colapso silencioso.

**Fórmula:**

$$\mathcal{L}_\text{KL} = \sum_{c=1}^{C} \max\!\Bigl(\lambda,\; \underbrace{\frac{1}{B \cdot d \cdot h \cdot w}\sum_{b,d,h,w} \frac{1}{2}\bigl(\mu_{c}^2 + e^{\sigma^2_{c}} - \sigma^2_{c} - 1\bigr)}_{\text{KL por canal}}\Bigr)$$

donde $\lambda$ = free\_bits (default 0.25) y la suma es sobre los $C$ canales latentes.

| Estado | Rango / señal | Acción |
|--------|--------------|--------|
| Sano ✓ | $\text{KL} \in [C\lambda,\; 3C]$ y creciendo | — |
| Aviso ⚠ | KL ≈ $C\lambda$ de forma persistente | Todos los canales clamped; posible collapse |
| Problema ✗ | KL = 0 o decrece a 0 | Posterior collapse total; reducir kl\_max\_beta |

**Por qué está aquí:** Si KL = 0 el encoder ignora la entrada y el espacio latente es inútil. Si KL es muy alto, el decoder recibe demasiado ruido. El free-bits de 0.25 garantiza que ningún canal desaparezca silenciosamente.

---

### `beta` — Peso dinámico del KL

**Qué mide:** El factor con que se multiplica $\mathcal{L}_\text{KL}$ en el total, siguiendo un warmup lineal.

**Fórmula:**

$$\beta(t) = \min\!\left(\frac{t}{T_\text{warmup}},\; 1\right) \cdot \beta_\text{max}$$

| Estado | Rango / señal | Acción |
|--------|--------------|--------|
| Sano ✓ | Rampa 0 → $\beta_\text{max}$ en $T_\text{warmup}$ steps, luego plateau | — |
| Problema ✗ | KL colapsa antes de que $\beta$ llegue a $\beta_\text{max}$ | Alargar kl\_warmup\_steps |

**Por qué está aquí:** Sin warmup, la regularización KL domina al inicio cuando el decoder aún no ha aprendido nada, colapsando el posterior antes de que haya utilidad.

---

### `kl_collapsed_fraction` — Fracción de canales colapsados

**Qué mide:** Proporción de canales latentes cuyo KL raw (pre-clamp) está por debajo del umbral free-bits, es decir, canales que el encoder ha «apagado».

**Fórmula:**

$$f_\text{collapsed} = \frac{1}{C}\sum_{c=1}^{C} \mathbf{1}[\text{KL}_c < \lambda]$$

| Estado | Rango / señal | Acción |
|--------|--------------|--------|
| Sano ✓ | Decrece hacia 0 con el entrenamiento | — |
| Aviso ⚠ | Se estanca en 0.3–0.7 | Algunos canales permanecen muertos; revisar beta |
| Problema ✗ | ≈ 1.0 de forma persistente | Colapso total; reducir kl\_max\_beta |

**Por qué está aquí:** Permite detectar cuántos canales están siendo «subvencionados» por el free-bits sin realmente codificar información.

---

## 2. Entrenamiento — train/

Logueadas en TensorBoard bajo `train/` cada `log_every` steps. Son ruidosas por naturaleza (un solo batch); usar `val/` para tendencias suaves.

---

### `train/kl_per_channel` (histograma)

**Qué mide:** Distribución del KL raw (pre-clamp) sobre los $C$ canales latentes en el batch actual.

**Fórmula:**

$$\text{KL}_c = \frac{1}{B \cdot d \cdot h \cdot w}\sum_{b,d,h,w} \frac{1}{2}\bigl(\mu_c^2 + e^{\sigma^2_c} - \sigma^2_c - 1\bigr)$$

| Estado | Rango / señal | Acción |
|--------|--------------|--------|
| Sano ✓ | Todos los canales con KL > free\_bits y dispersos | — |
| Aviso ⚠ | Mayoría de barras en KL ≈ 0 | Colapso parcial o total |
| Problema ✗ | Todo el histograma en KL = 0 | Colapso total; intervenir en beta/warmup |

**Por qué está aquí:** El histograma revela cuántos canales son activos y si la distribución es equilibrada. Más informativo que un escalar único.

---

### `train/mu_active_fraction`, `train/mu_n_active`

**Qué mide:** Fracción de canales latentes cuya media posterior $\mu$ varía de muestra a muestra (varianza > umbral). Un canal con $\text{Var}(\mu_c) \leq 0.01$ se considera «muerto» aunque su KL sea positivo.

**Fórmula:**

$$\text{AU} = \frac{1}{C}\sum_{c=1}^{C} \mathbf{1}\!\left[\widehat{\text{Var}}(\mu_c) > 0.01\right]$$

calculado sobre una ventana deslizante de 50 batches.

| Estado | Rango / señal | Acción |
|--------|--------------|--------|
| Sano ✓ | ≈ 1.0 (todos activos) | — |
| Aviso ⚠ | 0.5–0.8 | El modelo infrautiliza su capacidad; revisar beta |
| Problema ✗ | < 0.5 | Colapso parcial severo; intervenir |

**Por qué está aquí:** Complementa `kl_collapsed_fraction`: un canal puede tener KL > free\_bits porque $\sigma$ es grande, pero $\mu$ constante — lo que significa que no codifica información real de la entrada.

---

### `train/grad_norm` — Norma global del gradiente

**Qué mide:** La norma L2 del gradiente de todos los parámetros del generador antes del clipping.

**Fórmula:**

$$\|\nabla\| = \sqrt{\sum_{p}\sum_{i} \left(\frac{\partial \mathcal{L}}{\partial \theta_{p,i}}\right)^2}$$

| Estado | Rango / señal | Acción |
|--------|--------------|--------|
| Sano ✓ | 0.1–2.0 sin tendencia creciente | — |
| Aviso ⚠ | Picos esporádicos > 5 | Normal al inicio; vigilar si persisten |
| Problema ✗ | Crecimiento sostenido > 5 | Inestabilidad numérica; reducir lr o activar clipping |

**Por qué está aquí:** Es el indicador de salud del entrenamiento más inmediato. Calculado después de `scaler.unscale_()` para ser comparable independientemente de AMP.

---

### `train/grad_norm_encoder`, `train/grad_norm_decoder`, `train/grad_norm_mask_head`

**Qué mide:** Norma del gradiente por módulo, lo que permite localizar qué parte del modelo genera gradientes problemáticos.

| Estado | Rango / señal | Acción |
|--------|--------------|--------|
| Sano ✓ | Proporcionales entre sí | — |
| Aviso ⚠ | grad\_norm\_encoder >> grad\_norm\_decoder | Gradientes explosivos en encoder |
| Problema ✗ | grad\_norm\_mask\_head ≈ 0 | La cabeza de máscara no recibe gradiente; revisar mask\_total |

**Por qué está aquí:** La norma global no revela qué módulo es el problema. La descomposición por módulo permite diagnóstico quirúrgico.

---

### `train/grad_scaler_scale` — Escala del GradScaler AMP

**Qué mide:** El factor de amplificación que usa el GradScaler para evitar underflow en float16.

| Estado | Rango / señal | Acción |
|--------|--------------|--------|
| Sano ✓ | ~65536 (2¹⁶), estable | — |
| Aviso ⚠ | Caídas ocasionales | Normal si son esporádicas |
| Problema ✗ | Cae sostenidamente a 1–2 | Overflow de gradientes frecuente; reducir lr |

**Por qué está aquí:** Una escala que colapsa indica inestabilidad numérica en AMP que no sería visible en el grad\_norm (que ya viene tras unscale).

---

### `train/lr` — Learning rate

**Qué mide:** El learning rate actual del scheduler en el step actual.

**Por qué está aquí:** Verifica que el schedule esté configurado correctamente. Con `scheduler: none`, debe ser una línea horizontal en $\text{lr} = 2 \times 10^{-4}$.

---

### `train/steps_per_sec` — Throughput

**Qué mide:** Número de steps de entrenamiento por segundo, incluyendo forward, backward, eval de métricas de train, y logging.

| Estado | Rango / señal | Acción |
|--------|--------------|--------|
| Sano ✓ | 3–8 steps/s en GPU Ampere con batch\_size=4 | — |
| Aviso ⚠ | Caída sostenida > 20% | I/O bound o fragmentación de memoria |

**Por qué está aquí:** Detecta regresiones de rendimiento y cuellos de botella de I/O.

---

### `train/mask_pred_mean` — Media de la máscara predicha (entrenamiento)

**Qué mide:** La media de $\sigma(\hat{y})$ sobre el batch de entrenamiento actual. Es un detector de colapso de la cabeza de segmentación en tiempo real, sin esperar al eval.

**Fórmula:**

$$\bar{p} = \frac{1}{N}\sum_i \sigma(\hat{y}_i)$$

| Estado | Rango / señal | Acción |
|--------|--------------|--------|
| Sano ✓ | Converge hacia $\varphi_\text{mean} \approx 0.02$–$0.055$ | — |
| Aviso ⚠ | ≈ 0.5 sin moverse | La red no discrimina entre poro y sólido |
| Problema ✗ | → 0 o → 1 | Colapso total de la cabeza de máscara |

**Por qué está aquí:** Es el indicador más rápido de si la cabeza de máscara está aprendiendo, disponible en cada step.

---

### Métricas del discriminador (solo cuando `discriminator.enabled: true`)

---

#### `train/disc_loss` — Pérdida total del discriminador

**Qué mide:** Suma de las pérdidas LSGAN en muestras reales y falsas.

**Fórmula:**

$$\mathcal{L}_D = \frac{1}{2}\left[\mathbb{E}(D(x_\text{real})-1)^2 + \mathbb{E}\,D(x_\text{fake})^2\right]$$

| Estado | Rango / señal | Acción |
|--------|--------------|--------|
| Sano ✓ | Estable en 0.0–0.2 tras convergencia | — |
| Problema ✗ | → 0 rapidamente | D domina; G no aprende |

---

#### `train/d_loss_real`, `train/d_loss_fake` — Pérdida del discriminador descompuesta

**Qué mide:** `d_loss_real = 0.5 * E[(D(real)-1)²]` y `d_loss_fake = 0.5 * E[D(fake)²]` por separado.

**Por qué está aquí:** Si `d_loss_real` es alto pero `d_loss_fake` es bajo, D rechaza correctamente las falsas pero confunde las reales (probable problema de normalización). El inverso indica que G está engañando a D.

---

#### `train/gen_adv_loss` — Pérdida adversarial del generador

**Qué mide:** Con qué éxito el generador engaña al discriminador.

**Fórmula:**

$$\mathcal{L}_{G,\text{adv}} = \frac{1}{2}\mathbb{E}(D(\hat{x})-1)^2$$

| Estado | Rango / señal | Acción |
|--------|--------------|--------|
| Sano ✓ | Decrece gradualmente | — |
| Problema ✗ | No baja de 0.5 tras muchos steps | G no aprende a engañar; revisar disc\_weight |

---

#### `train/disc_score_real`, `train/disc_score_fake` — Puntuaciones continuas

**Qué mide:** La puntuación media asignada por D a muestras reales y falsas (sin umbralizar). LSGAN: targets son real→1, fake→0.

**Fórmula:**

$$s_\text{real} = \mathbb{E}[D(x_\text{real})], \quad s_\text{fake} = \mathbb{E}[D(\hat{x})]$$

| Estado | Rango / señal | Acción |
|--------|--------------|--------|
| Sano ✓ | $s_\text{real} \to 1$, $s_\text{fake} \to 0$ con el tiempo | — |
| Aviso ⚠ | $s_\text{real} < 0.5$ | D confunde las reales |
| Problema ✗ | $s_\text{fake} > 0.8$ | G engaña demasiado bien; D colapsado |

**Por qué está aquí:** Las accuracies binarias (>0.5) pierden la señal continua. $s_\text{real}=0.52$ y $s_\text{real}=0.95$ son muy diferentes pero producen la misma accuracy.

---

#### `train/disc_margin` — Margen de discriminación

**Qué mide:** La diferencia entre la puntuación media en reales y falsas. Es el indicador más directo de si el discriminador separa bien ambas distribuciones.

**Fórmula:**

$$\Delta = \mathbb{E}[D(x_\text{real})] - \mathbb{E}[D(\hat{x})]$$

| Estado | Rango / señal | Acción |
|--------|--------------|--------|
| Sano ✓ | Positivo y estable, ≈ 1.0 en convergencia | — |
| Aviso ⚠ | < 0.3 | D apenas separa; G ganando o D colapsando |
| Problema ✗ | → 0 o negativo | D ha colapsado; revisar disc\_weight o lr del discriminador |

**Por qué está aquí:** Es la señal más compacta del estado del entrenamiento adversarial. Un margen saludable garantiza que D da señal útil al generador.

---

#### `train/disc_acc_real`, `train/disc_acc_fake` — Accuracy binaria del discriminador

**Qué mide:** Fracción de muestras reales con $D > 0.5$ y fracción de falsas con $D < 0.5$.

**Por qué están aquí:** Complementan `disc_score_*` con una lectura de fácil interpretación intuitiva.

---

## 3. Validación — val/

Calculadas cada `eval_every=62` steps, promediando sobre `val_batches=100` batches. Curvas más suaves que `train/` y más representativas.

---

### `val/mae` — Mean Absolute Error XCT

**Qué mide:** Error de reconstrucción promedio vóxel a vóxel entre la XCT reconstruida y el ground truth, ambos en $[0,1]$.

**Fórmula:**

$$\text{MAE} = \frac{1}{N}\sum_i |\sigma(\hat{x}_i) - x_i|$$

| Estado | Rango / señal | Acción |
|--------|--------------|--------|
| Sano ✓ | < 0.02 | — |
| Aviso ⚠ | 0.02–0.05 | Reconstrucción mediocre; revisar xct\_weight |
| Problema ✗ | > 0.05 o no baja | El decoder no aprende textura XCT |

**Por qué está aquí:** Métrica de evaluación pura sin ponderaciones, comparable entre experimentos. Complementa la xct\_loss (Charbonnier en z-score).

---

### `val/sharpness_recon_over_gt` — Ratio de nitidez

**Qué mide:** Cuánta frecuencia espacial conserva la reconstrucción respecto al ground truth. Ratio de 1.0 = nitidez correcta. < 1.0 = borroneo. > 1.0 = artefactos de alta frecuencia.

**Fórmula:**

$$R_\text{sharp} = \frac{\bar{S}_\text{recon}}{\bar{S}_\text{gt}}, \quad S(x) = \frac{1}{3}\left(\overline{|\nabla_d x|} + \overline{|\nabla_h x|} + \overline{|\nabla_w x|}\right)$$

| Estado | Rango / señal | Acción |
|--------|--------------|--------|
| Sano ✓ | 0.9–1.1 | — |
| Aviso ⚠ | 0.7–0.9 | Borroneo moderado |
| Problema ✗ | < 0.7 crónico | Borroneo severo; el decoder promedia posibilidades |

**Por qué está aquí:** Un MAE bajo no implica nitidez — un volumen uniformemente borroso puede tener MAE aceptable pero gradientes espaciales mucho menores que el GT.

---

### `val/dice_pos_only` — Coeficiente Dice (parches no vacíos)

**Qué mide:** Solapamiento geométrico entre la máscara de poros predicha y el ground truth, excluyendo parches donde el GT es todo cero (que inflarían la media).

**Fórmula:**

$$\text{Dice} = \frac{2\,\text{TP} + \varepsilon}{2\,\text{TP} + \text{FP} + \text{FN} + \varepsilon}, \quad \varepsilon=10^{-7}$$

promediado sobre muestras con $\text{GT} > 0$.

| Estado | Rango / señal | Acción |
|--------|--------------|--------|
| Sano ✓ | > 0.7 | — |
| Aviso ⚠ | 0.5–0.7 | Prometedor para poros dispersos; seguir entrenando |
| Problema ✗ | < 0.3 | La red no está segmentando; revisar mask\_pred\_mean |

**Por qué está aquí:** Métrica de solapamiento estándar en segmentación médica. IoU y F1 han sido eliminados porque son transformaciones monotónicas de Dice para máscaras binarias.

---

### `val/precision_pos_only` — Precisión de segmentación

**Qué mide:** De todos los vóxeles que la red marca como poro, ¿qué fracción son realmente poro? Detecta falsos positivos (poros fantasma).

**Fórmula:**

$$\text{Precision} = \frac{\text{TP} + \varepsilon}{\text{TP} + \text{FP} + \varepsilon}$$

| Estado | Rango / señal | Acción |
|--------|--------------|--------|
| Sano ✓ | > 0.6, similar a recall | — |
| Aviso ⚠ | Precision << recall | Muchos falsos positivos; red muy agresiva |
| Problema ✗ | < 0.2 | Detecciones casi todas falsas |

---

### `val/recall_pos_only` — Recall de segmentación

**Qué mide:** De todos los poros reales, ¿qué fracción detecta la red? Detecta falsos negativos (poros perdidos).

**Fórmula:**

$$\text{Recall} = \frac{\text{TP} + \varepsilon}{\text{TP} + \text{FN} + \varepsilon}$$

| Estado | Rango / señal | Acción |
|--------|--------------|--------|
| Sano ✓ | > 0.6, similar a precision | — |
| Aviso ⚠ | Recall << precision | Muchos poros perdidos; Tversky $\beta$ insuficiente |
| Problema ✗ | < 0.2 | La red casi no detecta poros |

**Por qué están precision y recall aquí:** Descomponen el Dice: si Dice es bajo, la comparación precision vs recall dice si el fallo es por demasiados falsos positivos o negativos.

---

### `val/porosity_mae` — Error de porosidad ★ MÉTRICA PRINCIPAL

**Qué mide:** Error medio absoluto en la estimación de la fracción volumétrica de poros en cada parche. Esta es la **métrica de éxito primaria del proyecto**. El objetivo del LDM es generar volúmenes con la distribución de porosidad correcta.

**Fórmula:**

$$\text{Por-MAE} = \frac{1}{B}\sum_b \left|\frac{1}{N}\sum_i \sigma(\hat{y}_{b,i}) - \frac{1}{N}\sum_i y_{b,i}\right|$$

| Estado | Rango / señal | Acción |
|--------|--------------|--------|
| Sano ✓ | < 0.005 | Objetivo del proyecto cumplido |
| Aviso ⚠ | 0.005–0.015 | Aceptable en etapas tempranas |
| Problema ✗ | > 0.015 crónico | El decoder no estima porosidad; revisar mask\_bce pos\_weight |

**Por qué está aquí:** La segmentación espacialmente perfecta no es necesaria — lo que importa es estimar correctamente la cantidad de poros, que es lo que el pipeline de generación usa.

---

### `val/porosity_bias` — Sesgo de porosidad

**Qué mide:** Error signado medio en la estimación de porosidad. Positivo = el modelo sobreestima poros. Negativo = subestima.

**Fórmula:**

$$\text{Bias} = \frac{1}{B}\sum_b \left(\frac{\sum_i \sigma(\hat{y}_{b,i})}{N} - \frac{\sum_i y_{b,i}}{N}\right)$$

| Estado | Rango / señal | Acción |
|--------|--------------|--------|
| Sano ✓ | ≈ 0.0 | — |
| Aviso ⚠ | |Bias| > 0.003 con Por-MAE bajo | Sesgo sistemático que se cancela en el promedio |
| Problema ✗ | |Bias| > 0.01 | Sobreestimación/subestimación grave |

**Por qué está aquí:** Un MAE bajo puede enmascarar un bias si los errores positivos y negativos se cancelan. El bias revela si el modelo es sistemáticamente optimista o pesimista.

---

### `val/mask_pred_mean` — Media de la máscara predicha (validación)

Misma fórmula que `train/mask_pred_mean` pero sobre el conjunto de validación. Más estable y representativo.

| Estado | Rango / señal | Acción |
|--------|--------------|--------|
| Sano ✓ | ≈ $\varphi_\text{mean} \approx 0.02$–$0.055$ | — |
| Problema ✗ | → 0 | Colapso de la cabeza de máscara |

---

### `val/porosity_mae_bin_0` … `val/porosity_mae_bin_3` — MAE de porosidad por tramo

**Qué mide:** Error de porosidad stratificado por rango de GT-porosidad. Detecta si el modelo falla sistemáticamente en algún rango específico.

**Bins:** $[0, 1\%)$, $[1\%, 3\%)$, $[3\%, 6\%)$, $[6\%, \infty)$

| Bin | Tipo de parche | Target |
|-----|---------------|--------|
| bin\_0 [0–1%] | Casi sólido | Error < 0.002 |
| bin\_1 [1–3%] | Porosidad baja (más frecuente) | Error < 0.003 |
| bin\_2 [3–6%] | Porosidad media | Error < 0.005 |
| bin\_3 [6%+] | Alta porosidad (más difícil) | Error < 0.010 |

**Por qué está aquí:** El MAE global puede ocultar que el modelo falla solo en un rango. Si bin\_3 >> bin\_0, el modelo infravalora sistémicamente la alta porosidad.

---

### `val/mu_active_fraction`, `val/mu_n_active` — Canales latentes activos (validación)

Misma métrica que `train/mu_active_fraction` pero calculada sobre el conjunto de validación completo (acumulando momentos sobre todos los batches del eval). Más representativa que la ventana deslizante de train.

**Referencia:** Ver fórmula en §2.

---

### `val/mu_mean`, `val/mu_std` — Distribución de la media posterior

**Qué mide:** La media y desviación estándar de $\mu$ agregadas sobre el conjunto de validación.

| Estado | Señal | Acción |
|--------|-------|--------|
| `mu_mean` ≈ 0 | Prior correctamente centrada | — |
| `mu_mean` >> 0 | Offset sistemático del encoder | Riesgo al muestrear de la prior |
| `mu_std` ≈ 0 | Todos los parches producen el mismo código | Posible colapso |
| `mu_std` > 3 | Espacio latente no regularizado | Subir kl\_max\_beta |

**Por qué está aquí:** Verifica que el encoder usa el espacio latente de forma distribuida y centrada en la prior.

---

### `val/logvar_mean`, `val/std_mean` — Distribución de la varianza posterior

**Qué mide:** `logvar_mean` = media de $\log\sigma^2$; `std_mean` = media de $\sigma = e^{\frac{1}{2}\log\sigma^2}$.

| Estado | Señal | Interpretación |
|--------|-------|----------------|
| `logvar_mean` ≈ 0 | $\sigma \approx 1$ = prior | Balance sano |
| `logvar_mean` << −3 | $\sigma \approx 0.05$: muy determinista | Riesgo de actuar como autoencoder |
| `std_mean` < 0.1 | Red casi determinista | KL regularización insuficiente |

**Por qué está aquí:** Verifica que el VAE mantiene un posterior genuinamente estocástico.

---

### `val/kl` y `val/kl_raw`

`val/kl` = KL con free-bits aplicado (el que entra en la loss).
`val/kl_raw` = suma de KL por canal sin clampear.

Si `kl_raw` << `kl`, muchos canales están siendo subvencionados por el free-bits y en realidad están colapsados. Comparar ambos para detectar colapso latente.

---

## 4. Test — test/

Calculadas cada `test_every=625` steps sobre `test_batches=20` batches. Todas las métricas de §3 se replican bajo el prefijo `test/`. Las métricas adicionales exclusivas de test son:

---

### `test/porosity_mae_per_volume` — Histograma de error por volumen

**Qué mide:** Distribución del error de porosidad agregado por volumen completo (no por parche). Detecta si hay volúmenes específicos donde el modelo falla sistemáticamente.

**Cómo leerlo:** Histograma ideal: casi todo acumulado cerca de 0. Un pico secundario a la derecha indica un subgrupo de volúmenes con problemas.

---

### `test/porosity_mae_vol_p50`, `_p90`, `_max` — Percentiles de error por volumen

**Qué mide:** Error de porosidad a nivel de volumen completo en los percentiles 50, 90 y máximo.

| Métrica | Target | Alerta |
|---------|--------|--------|
| p50 | < 0.005 | — |
| p90 | < 0.010 | Subgrupo de volúmenes difíciles |
| max | < 0.020 | Si > 0.02, inspeccionar ese volumen manualmente |

**Por qué están aquí:** El p90 es más informativo que la media para robustez: un modelo con p50 excelente pero p90 malo falla en el 10% de los casos, lo que es inaceptable en la práctica.

---

## 5. Monte Carlo — montecarlo/

Calculado cada `montecarlo_every=100` steps sobre un batch fijo de `montecarlo_batch_size=8` parches. Realiza $N=30$ forward passes estocásticos (distinto $z$ muestreado en cada uno).

---

### Imágenes de incertidumbre Monte Carlo

**Qué miden:** La media y desviación estándar vóxel a vóxel de las 30 reconstrucciones. La std es un mapa de incertidumbre: zonas donde el modelo no sabe cuál es la salida correcta.

**Fórmula:**

$$\mu_\text{MC}(x) = \frac{1}{N}\sum_{n=1}^N f_\theta(x, z_n), \quad \sigma_\text{MC}(x) = \sqrt{\frac{1}{N}\sum_{n=1}^N (f_\theta(x, z_n) - \mu_\text{MC})^2}$$

| Tag | Contenido |
|-----|-----------|
| `montecarlo/xct_mean_{d,h,w}` | Media de 30 reconstrucciones XCT |
| `montecarlo/xct_std_{d,h,w}` | Incertidumbre XCT (colormap plasma) |
| `montecarlo/mask_mean_{d,h,w}` | Media de 30 predicciones de máscara |
| `montecarlo/mask_std_{d,h,w}` | Incertidumbre de máscara |
| `montecarlo/xct_recon_{d,h,w}` | Una sola reconstrucción (comparación con GT) |
| `montecarlo/xct_gt_{d,h,w}` | XCT ground truth |

**Cómo interpretarlos:**
- `xct_std` alta en bordes y poros = incertidumbre estructurada. **Sano.**
- `xct_std` alta en zonas homogéneas = ruido de muestreo sin información. kl\_max\_beta demasiado alto.
- `mask_std` uniforme = la cabeza de máscara ignora el latente.

**Nota:** Los escalares de diversidad (`xct_diversity`, `mask_diversity`) han sido eliminados — 30 forward passes → 1 número es una relación señal/coste muy baja. La información equivalente está en las imágenes de std. El colapso del reparametrización se detecta con un warning automático en el log.

---

## 6. Evaluación final completa

Calculada una sola vez al final del entrenamiento si `final_full_eval=true`. Pasa por **todos** los batches del loader (sin submuestra). Logs bajo `val_full/` y `test_full/`.

**Cómo usarla:** Filtrar en TensorBoard por `val_full` o `test_full`. Solo hay un único step registrado. **Estos son los valores a reportar como resultado definitivo del experimento.**

---

## 7. Evaluación offline

Ejecutada con `scripts/eval_checkpoint.py` sobre un checkpoint guardado. Reconstruye volúmenes completos ensamblando parches 64³ no solapados.

---

### `porosity_error` — Error de porosidad a escala de volumen completo

**Qué mide:** `|φ_pred - φ_gt|` calculado sobre el volumen completo (no por parche). Complementa `val/porosity_mae`.

**Fórmula:**

$$\varepsilon_\phi = \left|\frac{1}{|\Omega|}\sum_{v \in \Omega} \hat{y}_v - \frac{1}{|\Omega|}\sum_{v \in \Omega} y_v\right|$$

---

### `xct_mae` — MAE XCT a escala de volumen completo

**Qué mide:** Error de reconstrucción promedio sobre el volumen completo ensamblado.

---

### `xct_boundary_mae`, `mask_boundary_mae` — Consistencia de costuras ★ NUEVA

**Qué mide:** Discontinuidad media entre vóxeles adyacentes a ambos lados de las fronteras entre parches en el volumen ensamblado. Un valor bajo indica que el decoder produce reconstrucciones que se ensamblan sin saltos visibles.

**Fórmula:**

$$\text{Seam-MAE} = \frac{1}{|\mathcal{S}|}\sum_{(s^-, s^+) \in \mathcal{S}} \frac{1}{|F|}\sum_{v \in F} |v^-_{\text{last}} - v^+_{\text{first}}|$$

donde $\mathcal{S}$ es el conjunto de todas las caras de contacto entre parches adyacentes en la rejilla $64^3$ no solapada.

| Estado | Rango / señal | Acción |
|--------|--------------|--------|
| Sano ✓ | Cercano al error interno del volumen (< 0.01) | — |
| Aviso ⚠ | 2–3× mayor que el error interno medio | Costuras visibles; revisar blending |
| Problema ✗ | > 0.05 | Discontinuidades severas; considerar overlapping con cosine-taper |

**Por qué está aquí:** Parches reconstruidos de forma independiente pueden tener valores medios distintos en sus fronteras, creando costuras visibles en el volumen final. Esta métrica cuantifica ese artefacto.

---

### `s2_wasserstein` — Distancia Wasserstein de S₂(r)

**Qué mide:** Diferencia entre la función de correlación de dos puntos del volumen reconstruido y el ground truth. S₂(r) mide la probabilidad de que dos puntos separados una distancia $r$ estén ambos en fase porosa.

**Fórmula:**

$$W_1(S_2^\text{gt}, S_2^\text{pred}) = \inf_{\gamma \in \Pi} \mathbb{E}_{(r,s)\sim\gamma}|r-s|$$

donde las curvas se normalizan a distribuciones de probabilidad antes del cómputo.

| Estado | Rango / señal | Acción |
|--------|--------------|--------|
| Sano ✓ | Valor bajo, decreciente con el entrenamiento | — |
| Problema ✗ | Alto o creciente | La estructura porosa global no se conserva |

**Por qué está aquí:** Captura la estadística de largo alcance de la microestructura. Dos segmentaciones con el mismo Dice pueden tener S₂(r) muy distintas si una concentra los poros en clusters y la otra los dispersa.

---

### `psd_wasserstein` — Distancia Wasserstein de la distribución de tamaños de poro

**Qué mide:** Diferencia entre la distribución de volúmenes de poros individuales (componentes conectadas) del GT y la predicción.

**Por qué está aquí:** Complementa S₂(r) con información sobre el tamaño de poros individuales. Un modelo que acierte la fracción global pero que genere muchos poros pequeños en lugar de pocos grandes tendrá psd\_wasserstein alto.

---

### `memorization_nn_dist` — Puntuación de memorización

**Qué mide:** Distancia media al vecino más cercano en el espacio latente de cada parche de test con respecto a todos los parches de train. Valores bajos indican que el modelo ha memorizado el conjunto de entrenamiento.

**Fórmula:**

$$d_\text{mem} = \frac{1}{|Z_\text{test}|}\sum_{z \in Z_\text{test}} \min_{z' \in Z_\text{train}} \|z - z'\|_2$$

| Estado | Rango / señal | Acción |
|--------|--------------|--------|
| Sano ✓ | Similar a la distancia media entre parches de train | — |
| Problema ✗ | Muy bajo (< 50% del baseline train-train) | Posible memorización; revisar regularización |

---

### Morfología de poros (secundario)

`sphericity_mean` y `eq_diameter_mean`: caracterización de la forma y tamaño de los poros individuales. Útiles para comparaciones cualitativas en el paper pero secundarias para decisiones de entrenamiento.

---

## 8. Imágenes TensorBoard

Logueadas cada `image_log_every=62` steps bajo `val/` y en cada evaluación de test bajo `test/`.

| Tag | Contenido | Qué buscar |
|-----|-----------|-----------|
| `{prefix}/xct_gt_{d,h,w}` | XCT ground truth, corte central | Referencia |
| `{prefix}/xct_recon_{d,h,w}` | XCT reconstruida | Comparar bordes y textura con GT |
| `{prefix}/mask_gt_{d,h,w}` | Máscara GT (poros en blanco) | Referencia |
| `{prefix}/mask_recon_{d,h,w}` | Máscara predicha (continua, sin umbral) | Halos, poros fantasma, poros perdidos |

**Artefactos comunes:**

| Artefacto | Señal en imágenes | Métrica relacionada |
|-----------|-------------------|---------------------|
| Borroneo | recon más suave que GT | `sharpness_recon_over_gt` < 0.9 |
| Checkerboard | Patrón tablero en xct\_recon | `sharpness_recon_over_gt` > 1.1 |
| Poros fantasma | Manchas en mask\_recon sin GT | `precision_pos_only` baja |
| Poros perdidos | GT tiene poros ausentes en recon | `recall_pos_only` baja |
| Halo en máscara | Zona brillante alrededor de poros | FP en borde de poros |

---

## 9. Archivos en disco

| Archivo | Contenido | Cuándo |
|---------|-----------|--------|
| `log.jsonl` | Un JSON por step: todas las pérdidas train + val | Cada step + cada eval |
| `metrics.jsonl` | Solo registros val y test | Cada eval\_every / test\_every |
| `{name}_step{N:08d}.ckpt` | Checkpoint completo (model, optimizer, scaler, scheduler) | Cada save\_every=1000 steps |
| `samples/step_{N:08d}/{split}/` | Arrays 3D: xct\_gt, mask\_gt, xct\_recon, mask\_recon | Cada sample\_every=12500 steps |
| `eval_step{N:08d}.json` | Resultado completo de eval\_checkpoint.py | Por checkpoint evaluado |

---

## 10. Tabla de prioridades

| Criterio | Métrica | Target | Alerta | Acción |
|----------|---------|--------|--------|--------|
| **Éxito del proyecto** | `val/porosity_mae` | < 0.005 | > 0.015 | Revisar pos\_weight, mask\_bce |
| **Calidad XCT** | `val/mae` | < 0.02 | > 0.05 | Revisar xct\_weight, decoder |
| **Nitidez XCT** | `val/sharpness_recon_over_gt` | 0.9–1.1 | < 0.7 | Borroneo crónico: revisar KL |
| **Segmentación** | `val/dice_pos_only` | > 0.7 | < 0.3 | Revisar mask\_pred\_mean |
| **Sin colapso KL** | `kl_collapsed_fraction` | → 0 al final | ≈ 1.0 persistente | Reducir kl\_max\_beta |
| **Latente activo** | `val/mu_active_fraction` | ≈ 1.0 | < 0.5 | Revisar beta, free\_bits |
| **Sesgo porosity** | `val/porosity_bias` | ≈ 0.0 | \|bias\| > 0.01 | Revisar balance del dataset |
| **Robustez por volumen** | `test/porosity_mae_vol_p90` | < 0.010 | > 0.020 | Revisar splits y outliers |
| **Estabilidad train** | `train/grad_norm` | < 2.0 | > 5 sostenido | Reducir lr o activar clipping |
| **Consistencia de costuras** | `xct_boundary_mae` (offline) | < 0.01 | > 0.05 | Activar cosine-taper blending |
| **Adversarial (si activo)** | `train/disc_margin` | ≈ 1.0 | → 0 | Ajustar disc\_weight o lr\_disc |
| **Memorización** | `memorization_nn_dist` (offline) | > baseline | < 50% baseline | Aumentar regularización KL |
