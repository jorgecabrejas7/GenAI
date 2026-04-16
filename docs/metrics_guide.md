# Guía de Métricas — Pipeline de Entrenamiento VAE

Cubre **todas** las cantidades monitorizadas: pérdidas, métricas de reconstrucción, segmentación, espacio latente y Monte Carlo. Por cada métrica se indica la fórmula exacta (con referencia al archivo fuente), el razonamiento detrás de monitorizarla y cómo interpretar su curva.

---

## Índice

1. [Pérdidas](#1-pérdidas)
2. [Métricas de entrenamiento (train/)](#2-métricas-de-entrenamiento-train)
3. [Métricas de validación (val/)](#3-métricas-de-validación-val)
4. [Métricas de test (test/)](#4-métricas-de-test-test)
5. [Monte Carlo (montecarlo/)](#5-monte-carlo-montecarlo)
6. [Análisis offline (experiments/r03.py)](#6-análisis-offline-experimentsr03py)
7. [Evaluación final completa (val_full/ test_full/)](#7-evaluación-final-completa-val_full--test_full)
8. [Imágenes en TensorBoard](#8-imágenes-en-tensorboard)
9. [Archivos generados en disco](#9-archivos-generados-en-disco)
10. [Resumen de prioridades](#10-resumen-de-prioridades)

---

## 1. Pérdidas

Calculadas en cada step de train y de eval. Función principal: `compute_total_loss` en [losses/total.py](../src/poregen/losses/total.py).

---

### `total`

**Fórmula** ([losses/total.py:88-92](../src/poregen/losses/total.py)):
```
total = xct_weight × xct_loss + mask_total + beta × kl
```
Con R03: `xct_weight=1.0`, `mask_bce_weight=1.0`, `mask_dice_weight=1.0`.

**Razonamiento:** Es la única pérdida que llama a `.backward()`. Integra los tres objetivos del modelo: reconstrucción volumétrica (XCT), segmentación de poros (mask) y regularización del espacio latente (KL).

**Interpretación:** Debe descender de forma monótona y suave. Si val sube mientras train baja → overfitting. Oscilaciones fuertes → lr demasiado alto o batch size insuficiente.

---

### `xct_loss` — Reconstrucción XCT

**Fórmula** ([losses/recon.py:24-35](../src/poregen/losses/recon.py)):
```python
# Charbonnier (smooth L1), eps=1e-6, en espacio z-score
diff = pred_logits - target
xct_loss = mean( sqrt(diff² + eps²) )
```
Alternativas disponibles: `l1` y `mse` (seleccionables con `xct_loss_type`). R03 usa `charbonnier`.

**Razonamiento:** Charbonnier es diferenciable en 0 (a diferencia del L1 puro) pero se comporta como L1 para errores grandes, lo que lo hace más robusto que MSE ante outliers de intensidad. La pérdida opera en espacio z-score porque las salidas del decoder son ilimitadas y no se les aplica activación.

**Interpretación:** Valores < 0.01 son buenos. Si no baja de ~0.05, el decoder no recupera textura XCT. Si cae muy rápido mientras KL no sube, el modelo puede estar memorizando.

---

### `mask_bce` — Binary Cross-Entropy con pos_weight

**Fórmula** ([losses/mask.py:17-23](../src/poregen/losses/mask.py)):
```python
bce = mean( -[ pos_weight × y × log(σ(logit)) + (1-y) × log(1-σ(logit)) ] )
# R03: pos_weight = 17.16
```
Se usa `F.binary_cross_entropy_with_logits` (numéricamente estable).

**Razonamiento:** Sin `pos_weight`, la red aprende a predecir siempre cero y alcanza ~94.5% de accuracy trivialmente (la porosidad media es ~5.5%). Con `pos_weight ≈ (1-φ)/φ ≈ 17`, un falso negativo (poro no detectado) cuesta ~17× más que un falso positivo, equilibrando la contribución de ambas clases.

**Interpretación:** Una BCE que se estanca sin bajar más puede indicar que la red aprende la frecuencia global de poros pero no su forma ni posición.

---

### `mask_tversky` (o `mask_dice` si `use_tversky=false`)

**Fórmula** ([losses/mask.py:57-85](../src/poregen/losses/mask.py)):
```python
pred = sigmoid(logits)
pred_flat, target_flat = flatten spatial dims
tp = (pred_flat × target_flat).sum(dim=1)        # por muestra
fp = (pred_flat × (1 - target_flat)).sum(dim=1)
fn = ((1 - pred_flat) × target_flat).sum(dim=1)

tversky_coeff = (tp + 1) / (tp + alpha×fp + beta×fn + 1)
mask_tversky = 1 - mean(tversky_coeff)
# R03: alpha=0.3, beta=0.7  →  FN penaliza 2.3× más que FP
```
Si `use_tversky=false`, se usa Dice estándar (alpha=beta=0.5).

**Razonamiento:** La pérdida por región (Dice/Tversky) aporta gradiente a nivel de objeto, mientras que BCE da gradiente vóxel a vóxel. Con alpha=0.3, beta=0.7 se prioriza recall sobre precision, apropiado para poros pequeños y dispersos donde perderlos es más costoso que predecir alguno extra.

**Interpretación:** Rango [0,1]. Valor 0 = segmentación perfecta. Si baja más despacio que `mask_bce`, indica que la forma exacta de los poros es más difícil de aprender que su frecuencia global.

---

### `mask_total`

**Fórmula** ([losses/mask.py:126](../src/poregen/losses/mask.py)):
```python
mask_total = bce_weight × mask_bce + dice_weight × mask_tversky
# R03: bce_weight=1.0, dice_weight=1.0
```

**Razonamiento:** BCE sola converge a predecir la frecuencia base sin forma. Tversky sola tiene gradientes ruidosos al inicio cuando la red predice todo cero. Combinar ambas da señal complementaria y convergencia más estable.

---

### `kl` — Divergencia KL

**Fórmula** ([losses/kl.py:8-48](../src/poregen/losses/kl.py)):
```python
# Por elemento: forma (B, C, d, h, w)
kl_elem = 0.5 × (μ² + exp(logvar) - logvar - 1)

# Media sobre batch y dimensiones espaciales → (C,)
kl_per_channel = kl_elem.mean(dim=(0, 2, 3, 4))

# Free-bits: clamp mínimo por canal antes de sumar
if free_bits > 0:
    kl = clamp(kl_per_channel, min=free_bits).sum()
else:
    kl = kl_per_channel.sum()
# R03: free_bits=0.25
```

**Razonamiento:** Es la KL analítica entre la posterior del encoder `q(z|x) = N(μ,σ²)` y la prior `p(z) = N(0,I)`. Si es 0, el encoder ignora la entrada y el espacio latente es inútil (posterior collapse). Si es muy alto, el espacio latente está mal regularizado y la generación aleatoria produce ruido.

El free-bits de 0.25 garantiza que cada canal contribuya al menos ese KL antes de sumarse, evitando que canales individuales colapsen silenciosamente a la prior.

**Interpretación:** Con C=8 canales y free_bits=0.25, el mínimo esperado es 8 × 0.25 = 2.0 (todos clamped). Valores entre 2 y 10 son razonables con kl_max_beta=0.05. Si kl ≈ 2.0 de forma persistente y `kl_collapsed_fraction ≈ 1.0`, todos los canales están colapsados.

---

### `beta` — Peso dinámico del KL

**Fórmula** ([losses/kl.py:51-62](../src/poregen/losses/kl.py)):
```python
beta = min(step / kl_warmup_steps, 1.0) × kl_max_beta
# R03: kl_warmup_steps=4000, kl_max_beta=0.05
```

**Razonamiento:** El KL warmup da al decoder tiempo de aprender a reconstruir antes de que la regularización del espacio latente sea fuerte. Sin warmup, el término KL domina al inicio y colapsa el posterior.

**Interpretación:** Curva en TensorBoard: rampa lineal 0→0.05 durante 4000 steps, luego plateau en 0.05. Si el KL colapsa antes de que beta llegue a su máximo, considera alargar `kl_warmup_steps`.

---

### `kl_collapsed_fraction`

**Fórmula** ([losses/kl.py:42](../src/poregen/losses/kl.py)):
```python
kl_collapsed_fraction = (kl_per_channel < free_bits).float().mean()
# = fracción de canales con KL raw < 0.25
```

**Razonamiento:** Permite detectar cuántos canales están siendo "subvencionados" por el free-bits (su KL real es inferior al mínimo garantizado).

**Interpretación:** Al inicio del entrenamiento ≈ 1.0 (todos los canales clamped). Debe decrecer a medida que los canales aprenden a codificar información y su KL supera el umbral. Al final, `kl_collapsed_fraction ≈ 0.0` indica que todos los canales son activos. Si se mantiene ≈ 1.0, hay colapso total.

---

## 2. Métricas de entrenamiento (train/)

Logueadas en TensorBoard bajo el prefijo `train/` cada `log_every=1` steps (R03).

---

### `train/total`, `train/xct_loss`, `train/mask_bce`, `train/mask_tversky`, `train/kl`, `train/beta`, `train/kl_collapsed_fraction`

Versiones en tiempo real de las pérdidas descritas en §1. Son ruidosas por naturaleza (un solo batch). Para tendencias suaves, usar las curvas de `val/`.

---

### `train/kl_per_channel` (histograma)

El histograma del KL raw (pre-clamp) de los C canales latentes se loguea cada `log_every` steps bajo `train/kl_per_channel`. Los valores por canal individuales (`train/kl_ch{i}`) han sido eliminados para reducir ruido en TensorBoard.

**Interpretación:** Histograma ideal: todos los canales con KL > free_bits y bien distribuidos. Si la mayoría de barras están en KL ≈ 0, hay colapso parcial o total.

---

### `train/kl_active_channels`

**Fórmula** ([training/engine.py](../src/poregen/training/engine.py)):
```python
n_dead = round(kl_collapsed_fraction × len(kl_chs))
kl_active_channels = len(kl_chs) - n_dead
```

**Razonamiento:** Número entero de canales cuyo KL raw supera el umbral free_bits — más fácil de leer de un vistazo que la fracción.

**Interpretación:** Debe crecer desde 0 hasta C=8. Si se estanca en un valor bajo (ej. 2 de 8), el modelo infrautiliza su capacidad latente.

---

### `train/mu_active_fraction`, `train/mu_n_active`

**Fórmula** ([metrics/latent.py:64-79](../src/poregen/metrics/latent.py)):
```python
# Calculado sobre una ventana deslizante de los últimos 50 batches de train
mu_flat = mu.permute(1,0,...).reshape(C, -1)       # (C, N)
var_per_channel = sample_variance(mu_flat, dim=1)  # (C,) — varianza de μ
mu_n_active = (var_per_channel > threshold).sum()  # threshold=0.01
mu_active_fraction = mu_n_active / C
```
El motor recomputa esto cada 10 steps sobre una deque de 50 momentos.

**Razonamiento:** Un canal "activo" (μ-varianza) es aquel cuya media μ varía de muestra a muestra (varianza > 0.01). Complementa `kl_collapsed_fraction`: un canal puede tener KL > free_bits pero varianza de μ baja si la std σ hace todo el trabajo.

**Interpretación:** Idealmente `mu_active_fraction ≈ 1.0`. La ventana deslizante de 50 batches da una estimación suavizada sin coste de computar sobre todo el dataset.

---

### `train/grad_norm`

**Fórmula** ([training/engine.py:147-150](../src/poregen/training/engine.py)):
```python
grad_norm = torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_grad_norm if max_grad_norm is not None else float("inf")
).item()
# R03: max_grad_norm=null → sin clipping; grad_norm es solo observación
```

**Razonamiento:** La norma del gradiente es el indicador de salud del entrenamiento más inmediato. Se calcula **después** de `scaler.unscale_()` para que sea comparable independientemente del GradScaler.

**Interpretación:** Con R03 (sin clipping), valores entre 0.1 y 2.0 son normales. Picos esporádicos son esperables al inicio. Un grad_norm que crece de forma sostenida indica inestabilidad numérica o lr demasiado alto. Si se activa clipping (`max_grad_norm` distinto de null), valores constantemente en el límite indican que el clipping está activo en cada step.

---

### `train/grad_norm_encoder`, `train/grad_norm_decoder`, `train/grad_norm_mask_head`

**Fórmula** ([training/engine.py](../src/poregen/training/engine.py)):
```python
# Computed after scaler.unscale_(), before global clip, with max_norm=inf
grad_norm_encoder  = clip_grad_norm_(model.encoder.parameters(),   float("inf"))
grad_norm_decoder  = clip_grad_norm_(model.decoder.parameters(),   float("inf"))
grad_norm_mask_head= clip_grad_norm_(model.mask_head.parameters(), float("inf"))
```

**Razonamiento:** Descomponer la norma global por módulo permite detectar qué parte del modelo genera gradientes explosivos o muertos, algo que el `grad_norm` global no revela.

**Interpretación:** Si `grad_norm_encoder` es muy alto pero `grad_norm_decoder` es normal, el cuello de botella es el encoder. Si `grad_norm_mask_head ≈ 0`, la cabeza de máscara no recibe gradiente (posible bug o colapso de la loss de máscara).

---

### `train/grad_scaler_scale`

**Fórmula** ([training/engine.py](../src/poregen/training/engine.py)):
```python
scaler.get_scale()   # escala de amplificación del GradScaler AMP
```

**Razonamiento:** El GradScaler ajusta dinámicamente el factor de escala para evitar underflow en float16. Si la escala cae repetidamente (lo que indica overflow de gradientes), el entrenamiento está numéricamente inestable.

**Interpretación:** Valor típico: 65536 (2¹⁶). Si cae de forma sostenida a 1 o 2, hay overflow de gradientes frecuente — considerar reducir lr o max_grad_norm.

---

### `train/mask_pred_mean`

**Fórmula** ([training/engine.py](../src/poregen/training/engine.py)):
```python
train/mask_pred_mean = sigmoid(mask_logits).mean()
```

**Razonamiento:** Detector en tiempo real de colapso de la cabeza de segmentación durante entrenamiento, sin esperar al eval.

**Interpretación:** Debe converger a ≈ φ_mean (~0.02-0.055). Si ≈ 0 → la red predice siempre "no poro". Si ≈ 0.5 → no discrimina. Idéntica interpretación a `val/mask_pred_mean` pero disponible en cada step.

---

### `train/steps_per_sec`

**Fórmula** ([training/engine.py](../src/poregen/training/engine.py)):
```python
steps_per_sec = 1.0 / time.perf_counter()  # medido alrededor del train_step completo
```

**Razonamiento:** Métrica de throughput para detectar cuellos de botella de I/O, regresiones de rendimiento tras cambios de código, o diferencias entre hardware.

**Interpretación:** Con batch_size=4 y patches 64³, valores típicos de 3-8 steps/s en GPU Ampere. Caídas sostenidas indican I/O-bound (prefetch insuficiente) o fragmentación de memoria.

---

### `train/lr`

**Fórmula:** Learning rate actual del scheduler.

R03 usa `scheduler: none` y `warmup_steps: 0` → lr fijo en 2e-4 durante todo el entrenamiento. La curva debe ser una línea horizontal. Si ves una rampa o coseno, la configuración no corresponde a R03.

---

## 3. Métricas de validación (val/)

Calculadas cada `eval_every=62` steps, promediando sobre `val_batches=100` batches (R03). Prefijo `val/` en TensorBoard.

---

### `val/mae` — Mean Absolute Error XCT

**Fórmula** ([metrics/recon.py:22-24](../src/poregen/metrics/recon.py)):
```python
pred = sigmoid(xct_logits)          # → [0, 1]
mae  = F.l1_loss(pred, xct_target)  # ambos en [0, 1]
```
Se aplica `sigmoid` a los logits del decoder (salida sin activación) antes de la comparación, de modo que ambos tensores están en [0, 1] (XCT normalizado a uint8/255).

**Razonamiento:** Métrica de evaluación pura sin ponderaciones, comparable entre experimentos. Complementa la loss de entrenamiento (Charbonnier en z-score).

**Interpretación:** Rango [0,1]. MAE < 0.02 es bueno; < 0.01 es excelente. Si MAE es bajo pero las imágenes parecen borrosas, mirar también `sharpness_recon`.

---

### `val/psnr` — Peak Signal-to-Noise Ratio

**Fórmula** ([metrics/recon.py:28-37](../src/poregen/metrics/recon.py)):
```python
pred = sigmoid(xct_logits)                   # → [0, 1]
mse  = F.mse_loss(pred, xct_gt)
psnr = 10 × log₁₀(1.0 / mse)               # max_val=1.0 porque ambos en [0,1]
```
Se aplica `sigmoid` antes del cómputo para asegurar que ambos tensores están en [0, 1].

**Razonamiento:** Escala logarítmica que comprime el rango del MSE en una cifra legible. Estándar en compresión de imagen y reconstrucción volumétrica.

**Interpretación:** Cada +3 dB = error reducido a la mitad. > 30 dB = buena calidad; > 35 dB = excelente. PSNR alto no garantiza nitidez (puede haber borroneo uniforme con PSNR aceptable).

---

### `val/sharpness_recon`, `val/sharpness_recon_over_gt`

**Fórmula** ([metrics/recon.py:41-56](../src/poregen/metrics/recon.py)):
```python
# Calculado sobre sigmoid(xct_logits) y xct_gt, ambos en [0,1]
gd = (x[:,:,1:,:,:] - x[:,:,:-1,:,:]).abs().mean()   # gradiente en D
gh = (x[:,:,:,1:,:] - x[:,:,:,:-1,:]).abs().mean()   # gradiente en H
gw = (x[:,:,:,:,1:] - x[:,:,:,:,:-1]).abs().mean()   # gradiente en W
sharpness = (gd + gh + gw) / 3.0
```

El ratio se calcula como ([training/engine.py](../src/poregen/training/engine.py)):
```python
sharpness_recon_over_gt = sharpness_recon / sharpness_gt
```

**Razonamiento:** La magnitud del gradiente espacial es un proxy de la frecuencia espacial del volumen. Un modelo que promedia posibilidades genera reconstrucciones con gradientes más suaves que el GT.

**Interpretación:**
- `sharpness_recon` debe converger hacia el valor de referencia del GT.
- `sharpness_recon_over_gt ≈ 1.0` = nitidez correcta.
- `< 1.0` = borroneo (falta alta frecuencia).
- `> 1.0` = artefactos de alta frecuencia (ruido, checkerboard).

**Nota:** `val/sharpness_gt` se loguea **una sola vez** en TensorBoard (step 0) calculado sobre el primer batch del primer eval — es una constante del dataset, no del modelo, y recomputarla cada eval sería ruido.

---

### `val/recon_xct_mean`

**Fórmula** ([training/engine.py](../src/poregen/training/engine.py)):
```python
recon_xct_mean = sigmoid(output.xct_logits).mean()   # → [0, 1]
```

**Razonamiento:** Un sesgo sistemático en el nivel medio de intensidad indica que el decoder no está calibrado con el target. Se aplica `sigmoid` para que el valor sea comparable con las targets normalizadas /255.

**Interpretación:** Debe estar próxima a ~0.5 (media global de las targets XCT normalizadas a uint8/255). Una desviación grande indica sesgo: el decoder predice intensidades sistemáticamente más altas o más bajas.

---

### `val/recon_xct_std`

**Fórmula** ([training/engine.py](../src/poregen/training/engine.py)):
```python
recon_xct_std = sigmoid(output.xct_logits).std()   # → [0, 1]
```

**Razonamiento:** Una std muy baja indica predicciones "planas" donde el decoder promedia posibilidades en lugar de comprometerse con una textura concreta. Se aplica `sigmoid` para comparabilidad con el espacio de targets.

**Interpretación:** Si es mucho menor que la std del GT, el espacio latente no tiene suficiente información para reconstruir detalles. Señal directa de posterior collapse o capacidad latente insuficiente.

---

### `val/dice_pos_only`

**Fórmula** ([metrics/seg.py:57](../src/poregen/metrics/seg.py)):
```python
pred = (sigmoid(logits) >= 0.5).float()
# Por muestra b:
tp = (pred[b] × target[b]).sum()
fp = (pred[b] × (1-target[b])).sum()
fn = ((1-pred[b]) × target[b]).sum()
dice[b] = (2×tp + 1e-7) / (2×tp + fp + fn + 1e-7)

# dice_pos_only: media sobre los b donde target[b].sum() > 0
```

**Razonamiento:** Dice mide solapamiento geométrico a nivel de parche. `_pos_only` excluye parches donde el GT es todo cero — la variante `_all` estaba inflada artificialmente y ha sido eliminada.

**Interpretación:** Dice > 0.5 = prometedor; > 0.7 = bueno para poros pequeños y dispersos.

---

### `val/iou_pos_only`

**Fórmula** ([metrics/seg.py:58](../src/poregen/metrics/seg.py)):
```python
iou[b] = (tp + 1e-7) / (tp + fp + fn + 1e-7)
```

**Razonamiento:** Más estricto que Dice porque el denominador es mayor (IoU = Dice / (2 - Dice)). Dice=0.7 corresponde a IoU≈0.54.

**Interpretación:** Métrica complementaria a `dice_pos_only`.

---

### `val/precision_all`, `val/precision_pos_only`

**Fórmula** ([metrics/seg.py:87](../src/poregen/metrics/seg.py)):
```python
# Agregado sobre todos los b del subconjunto seleccionado
precision = (TP_total + 1e-7) / (TP_total + FP_total + 1e-7)
```

**Razonamiento:** "De todos los vóxeles que la red marca como poro, ¿qué fracción son realmente poro?" Detecta si la red produce demasiados poros fantasma.

**Interpretación:** Precision alta con recall bajo = red conservadora. Precision baja = muchos falsos positivos (poros fantasma).

---

### `val/recall_all`, `val/recall_pos_only`

**Fórmula** ([metrics/seg.py:88](../src/poregen/metrics/seg.py)):
```python
recall = (TP_total + 1e-7) / (TP_total + FN_total + 1e-7)
```

**Razonamiento:** "De todos los poros reales, ¿qué fracción detecta la red?" Dado que la pérdida Tversky penaliza FN más que FP (beta=0.7 > alpha=0.3), el entrenamiento está sesgado hacia mejorar el recall.

**Interpretación:** Recall bajo = la red pierde muchos poros. Recall alto + precision baja = la red es muy agresiva.

---

### `val/f1_pos_only`

**Fórmula** ([metrics/seg.py:89](../src/poregen/metrics/seg.py)):
```python
f1 = 2 × precision × recall / (precision + recall + 1e-7)
```

**Razonamiento:** Media armónica de precision y recall. Calculado sobre TP/FP/FN acumulados del batch (agregado), a diferencia de Dice que es media de ratios por muestra. La variante `_all` ha sido eliminada por inflación estadística.

**Interpretación:** Úsalo para confirmar consistencia con Dice. Si f1 >> dice_pos_only o viceversa, puede haber muestras extremas sesgando uno de los dos cálculos.

---

### `val/porosity_mae` — Error de Porosidad (Métrica Principal)

**Fórmula** ([metrics/seg.py:101-129](../src/poregen/metrics/seg.py)):
```python
pred_por = sigmoid(mask_logits).mean(dim=(1,2,3,4))  # (B,) — porosidad predicha por parche
gt_por   = mask_target.mean(dim=(1,2,3,4))            # (B,) — porosidad real por parche
signed   = pred_por - gt_por
porosity_mae = signed.abs().mean()
```

**Razonamiento:** Esta es la **métrica de éxito primaria del proyecto**. El objetivo final no es segmentar poros perfectamente vóxel a vóxel, sino estimar correctamente la fracción volumétrica de poros en cada parche (lo que se usará en el pipeline de generación latente).

**Interpretación:** Target: < 0.005 (0.5% de volumen). Si la porosidad media es φ_mean ≈ 0.02, un error de 0.005 significa que la red puede predecir entre 1.5% y 2.5% cuando la realidad es 2%. Nota: esta métrica puede ser baja incluso si la segmentación espacial es imperfecta — la red puede acertar la cantidad total pero tener los poros en sitios equivocados.

---

### `val/porosity_bias`

**Fórmula** ([metrics/seg.py:127](../src/poregen/metrics/seg.py)):
```python
porosity_bias = signed.mean()    # = mean(pred_por - gt_por)
```

**Razonamiento:** Detecta sesgo sistemático que un MAE bajo podría enmascarar.

**Interpretación:** Positivo = sobreestima porosidad. Negativo = subestima. Si MAE es bajo pero bias es alto, el modelo tiene un sesgo que se cancela en el promedio — señal de que algo está mal en las predicciones individuales.

---

### `val/mask_pred_mean`

**Fórmula** ([metrics/seg.py:128](../src/poregen/metrics/seg.py)):
```python
mask_pred_mean = sigmoid(mask_logits).mean()
```

**Razonamiento:** Detector de collapse de la cabeza de segmentación.

**Interpretación:** Debe converger a ≈ φ_mean (fracción de poros real del dataset, ~0.02-0.055). Si ≈ 0 → la red no predice ningún poro. Si ≈ 0.5 → la red no discrimina. Es el indicador más rápido de si la cabeza de máscara está aprendiendo.

---

### `val/porosity_mae_bin_0` … `val/porosity_mae_bin_3`

**Fórmula** ([metrics/seg.py](../src/poregen/metrics/seg.py)):
```python
# Bins por GT-porosidad del parche: [0, 0.01), [0.01, 0.03), [0.03, 0.06), [0.06+)
pred_por = sigmoid(mask_logits).mean(dim=(1,2,3,4))   # (N,) — acumulado sobre eval
gt_por   = mask_target.mean(dim=(1,2,3,4))             # (N,)
abs_err  = |pred_por - gt_por|

porosity_mae_bin_i = abs_err[gt_por in bin_i].mean()
```
Calculado sobre todos los parches de los `n_batches` del eval concatenados, no batch a batch.

**Razonamiento:** El MAE global de porosidad puede enmascarar que la red falla sistemáticamente en rangos específicos de porosidad. Los parches casi vacíos (bin_0) y los más porosos (bin_3) suelen ser los más difíciles.

**Interpretación:**
- `bin_0` [0–1%]: parches casi sólidos — error suele ser muy bajo.
- `bin_1` [1–3%]: porosidad baja — el rango más frecuente; error objetivo < 0.003.
- `bin_2` [3–6%]: porosidad media — zona de transición.
- `bin_3` [6%+]: alta porosidad — los parches más porosos, suelen ser los más difíciles.
Si `bin_3` tiene MAE muy superior al resto, el modelo infravalora sistémicamente la alta porosidad.

---

### `val/mu_active_fraction`, `val/mu_n_active`

**Fórmula** ([metrics/latent.py:64-79](../src/poregen/metrics/latent.py)):
```python
# Calculado sobre los n_batches del eval, acumulando momentos
mu_flat = mu.permute(1,0,...).reshape(C, -1)       # (C, N) donde N = B×d×h×w acumulados
var_per_channel = sample_variance(mu_flat, dim=1)  # (C,) — varianza de μ
mu_n_active = (var_per_channel > 0.01).sum()
mu_active_fraction = mu_n_active / C
```

**Razonamiento:** Mide cuántos canales del espacio latente codifican información real que varía de entrada en entrada (métrica basada en var(μ)). Canales con varianza de μ baja son "muertos". Complementa `kl_collapsed_fraction`: un canal puede tener KL > free_bits pero varianza de μ baja (la std σ hace todo el trabajo pero μ es constante).

**Interpretación:** Idealmente `mu_active_fraction = 1.0` (todos los canales activos). Si < 0.5, el modelo está infrautilizando su capacidad.

---

### `val/mu_mean`

**Fórmula** ([metrics/latent.py:127](../src/poregen/metrics/latent.py)):
```python
mu_mean = mu.mean()
```

**Razonamiento:** La prior es N(0,I), así que μ debería tener media ≈ 0 si la regularización KL es efectiva.

**Interpretación:** Un valor muy alejado de 0 indica que el encoder usa un offset sistemático, causando problemas si se muestrea de la prior durante generación (el muestreo produciría distribuciones fuera del rango de entrenamiento del decoder).

---

### `val/mu_std`

**Fórmula** ([metrics/latent.py:128](../src/poregen/metrics/latent.py)):
```python
mu_std = mu.std()
```

**Razonamiento:** Mide la dispersión global de las medias posteriores. Si ≈ 0, todos los parches producen el mismo código latente (sin información). Si es muy alta, el espacio latente no está regularizado.

**Interpretación:** Valores entre 0.5 y 2.0 suelen indicar un encoder activo y regularizado.

---

### `val/logvar_mean`

**Fórmula** ([metrics/latent.py:129](../src/poregen/metrics/latent.py)):
```python
logvar_mean = logvar.mean()
```

**Razonamiento:** Un logvar=0 corresponde a σ=1 (igual que la prior). Se monitoriza para detectar dos problemas opuestos: posterior demasiado determinista o demasiado difuso.

**Interpretación:** Valores muy negativos (ej. -5) → σ ≈ 0.08: red muy determinista, alejada de la prior, riesgo de actuar como autoencoder sin regularización. Valores positivos altos → incertidumbre excesiva. Balance sano: cercano a 0.

---

### `val/logvar_std`

**Fórmula** ([metrics/latent.py:130](../src/poregen/metrics/latent.py)):
```python
logvar_std = logvar.std()
```

**Razonamiento:** Una std alta del logvar indica que el encoder asigna distintos niveles de certeza a distintos parches, lo cual es deseable (zonas homogéneas = mayor certeza que zonas complejas).

**Interpretación:** Std baja → el encoder trata todos los parches con la misma incertidumbre, sin distinción. Std alta → comportamiento adaptativo correcto.

---

### `val/std_mean`

**Fórmula** ([metrics/latent.py:131](../src/poregen/metrics/latent.py)):
```python
std_mean = exp(0.5 × logvar).mean()   # = mean(σ)
```

**Razonamiento:** Complementa `logvar_mean` con una lectura directa en escala de desviación estándar (más intuitiva que el espacio logarítmico).

**Interpretación:** Al inicio del entrenamiento (antes del warmup KL) debe ser ≈ 1.0. Con el entrenamiento puede bajar moderadamente. Si `std_mean` < 0.1, la red opera casi como autoencoder determinista y la regularización KL es insuficiente.

---

### `val/kl` (y `val/kl_ch00` ... `val/kl_ch{C-1}`)

Los mismos valores que en train pero promediados sobre 100 batches de validación. Más estables y representativos que las curvas de train. También se loguea el histograma `val/kl_per_channel`.

### `val/kl_raw`

**Fórmula** ([training/engine.py](../src/poregen/training/engine.py)):
```python
kl_raw = sum(kl_per_channel)   # suma del vector (C,) raw, sin clamp de free-bits
```

**Razonamiento:** Permite comparar la KL pre-clamp del encoder con la KL efectiva de la loss (post free-bits). Si `kl_raw` << `kl` (con clamp), muchos canales están siendo subvencionados y en realidad colapsados.

**Interpretación:** Compara con `kl` (con clamp) para detectar cuántos canales son realmente inactivos.

---

## 4. Métricas de test (test/)

Calculadas cada `test_every=625` steps, promediando sobre `test_batches=20` batches (R03). Prefijo `test/` en TensorBoard.

Todas las métricas de §3 se replican exactamente con las mismas fórmulas e interpretaciones bajo el prefijo `test/` (incluyendo `test/porosity_mae_bin_{0-3}`). Las métricas adicionales exclusivas de test son:

---

### `test/porosity_mae_per_volume` — Histograma de error por volumen

**Fórmula** ([training/engine.py](../src/poregen/training/engine.py)):
```python
# Durante el eval, se acumula el error signado por volume_id:
vol_por_errors[volume_id].append( (pred_por_patch - gt_por_patch).item() )

# Al final, por cada volumen:
volume_mae = abs( mean(signed_errors_for_this_volume) )

# Se loguea como histograma en TensorBoard
tb_writer.add_histogram("test/porosity_mae_per_volume", per_vol_maes, step)
```

**Razonamiento:** Detecta si hay volúmenes concretos donde la red falla sistemáticamente, cosa que el MAE global no revelaría (errores de distintos volúmenes se cancelan entre sí).

**Interpretación:** Histograma ideal: casi todas las barras cerca de 0 con cola corta. Si aparece un pico secundario a la derecha (ej. en 0.02), hay un subgrupo de volúmenes con porosidades sistemáticamente mal predichas. Puede indicar: (a) tipo de muestra no visto en train, (b) grupo específico del dataset con características distintas, (c) artefacto de normalización específico de esos volúmenes.

---

### `test/porosity_mae_vol_p50`, `test/porosity_mae_vol_p90`, `test/porosity_mae_vol_max`

**Fórmula** ([training/engine.py](../src/poregen/training/engine.py)):
```python
per_vol_maes = tensor([abs(mean(errs)) for errs in vol_por_errors.values()])
test/porosity_mae_vol_p50 = per_vol_maes.quantile(0.5)   # mediana
test/porosity_mae_vol_p90 = per_vol_maes.quantile(0.9)   # percentil 90
test/porosity_mae_vol_max = per_vol_maes.max()            # peor volumen
```

**Razonamiento:** El histograma `porosity_mae_per_volume` resume la distribución visualmente pero no produce un escalar que se pueda vigilar en alertas o comparar entre experimentos. Los percentiles dan puntos de comparación directos.

**Interpretación:**
- `p50` ≈ MAE del volumen mediano — similar al MAE global si la distribución es simétrica.
- `p90` — el 10% peor de los volúmenes; debe estar por debajo de 0.01 para considerar el modelo robusto.
- `max` — el peor caso absoluto; cualquier volumen con MAE > 0.02 merece inspección manual.

---

## 5. Monte Carlo (montecarlo/)

Calculado cada `montecarlo_every=100` steps (R03) sobre un batch fijo de `montecarlo_batch_size=8` parches capturado del primer batch de entrenamiento. Realiza N=30 forward passes estocásticos (diferente z muestreado en cada uno, usando la reparametrización del VAE).

**Razonamiento:** El VAE tiene incertidumbre epistémica intrínseca: dada la misma entrada, cada forward pass produce una reconstrucción diferente porque `z = μ + σ × ε` con ε ~ N(0,I). La variabilidad entre las 30 reconstrucciones cuantifica cuánta incertidumbre tiene el modelo sobre la salida correcta.

---

### `montecarlo/xct_diversity`, `montecarlo/mask_diversity`

**Fórmula** ([training/engine.py:768-779](../src/poregen/training/engine.py)):
```python
xct_stack  = stack(N forward passes)              # (N, B, 1, D, H, W)
mask_stack = stack(N forward passes)

xct_std  = xct_stack.std(dim=0)                   # (B, 1, D, H, W)
mask_std = mask_stack.std(dim=0)

xct_diversity  = xct_std.mean().item()            # escalar global
mask_diversity = mask_std.mean().item()
```
Se loguea una advertencia si alguno es < 1e-5 (indica colapso del reparametrización).

**Razonamiento:** Una diversidad ≈ 0 significa que el encoder está produciendo distribuciones posteriores muy estrechas (σ → 0), lo que equivale a un autoencoder determinista: el VAE ha dejado de ser generativo. Una diversidad excesiva indica que el decoder no es suficientemente determinista dado z.

**Interpretación:**
- `xct_diversity` ≈ 0 → colapso del reparametrización, el modelo es determinista, la generación desde prior no funcionará.
- `mask_diversity` ≈ 0 → la cabeza de máscara ignora z (predice siempre lo mismo independientemente del latente).
- Valores > 0.01 indican que el VAE mantiene estocasticidad real.

---

### Imágenes Monte Carlo

**Fórmula** ([training/engine.py:789-812](../src/poregen/training/engine.py)):
```python
xct_mean  = xct_stack.mean(dim=0)     # (B,1,D,H,W) — media de N=30 forward passes
xct_std   = xct_stack.std(dim=0)      # (B,1,D,H,W) — std de N=30 forward passes
mask_mean = mask_stack.mean(dim=0)
mask_std  = mask_stack.std(dim=0)
```

Se logean cortes centrales en los tres ejes (d=axial, h=coronal, w=sagittal):

| Tag TensorBoard | Contenido |
|---|---|
| `montecarlo/xct_mean_{d,h,w}` | Media de 30 reconstrucciones XCT (grayscale) |
| `montecarlo/xct_std_{d,h,w}` | Std vóxel a vóxel de las 30 reconstrucciones XCT (colormap plasma) |
| `montecarlo/mask_mean_{d,h,w}` | Media de 30 predicciones de máscara (grayscale) |
| `montecarlo/mask_std_{d,h,w}` | Std vóxel a vóxel de las 30 predicciones de máscara (colormap plasma) |
| `montecarlo/xct_recon_{d,h,w}` | Una sola reconstrucción (primer sample) para comparar con GT |
| `montecarlo/xct_gt_{d,h,w}` | XCT ground truth del batch fijo |
| `montecarlo/mask_recon_{d,h,w}` | Una sola predicción de máscara |
| `montecarlo/mask_gt_{d,h,w}` | Máscara ground truth del batch fijo |

**Interpretación:**
- `xct_std` alta en bordes y poros = incertidumbre estructurada (el modelo sabe dónde no está seguro). Esto es **sano**.
- `xct_std` alta en zonas homogéneas = ruido de muestreo sin información. Indica kl_max_beta demasiado alto o espacio latente sobre-regularizado.
- `mask_std` alta justo en los bordes de poros = incertidumbre en localización exacta. Normal.
- `mask_std` uniforme por todo el volumen = la cabeza de máscara no está usando el latente.

---

## 6. Análisis offline (experiments/r03.py)

Conjunto de utilidades en [experiments/r03.py](../src/poregen/experiments/r03.py) diseñadas para análisis post-hoc en notebooks. No generan TensorBoard ni ficheros durante el entrenamiento — se usan con un checkpoint cargado.

---

### `VAEOutput.z` — Latente muestreado

**Fórmula** ([models/vae/base.py:66](../src/poregen/models/vae/base.py)):
```python
z = mu + exp(0.5 * logvar) * eps    # eps ~ N(0, I)   [reparametrización]
```
Campo `z` del `VAEOutput`. El decoder usa `z`, no `mu`.

**Razonamiento:** Durante el entrenamiento, el decoder siempre recibe `z` muestreado. Para análisis determinista (ej. construir un espacio latente para PCA, clustering, o una pipeline de difusión) se usa `encode_mu` que devuelve solo `mu` sin ruido.

**Diferencia con `mu`:** `z` = `mu` + ruido. Las métricas de entrenamiento (`mu_active_fraction`, `latent_stats`) solo monitorizan `mu` — razonable porque `mu` captura la información semántica y el ruido de `z` se promedia a cero en el agregado. Para verificar compatibilidad con una prior N(0,I) para difusión, habría que rastrear también la distribución de `z` (ver nota al pie de §3).

---

### Decoder auxiliar — `AuxiliaryXCTDecoder`

**Qué es** ([experiments/r03.py:284-324](../src/poregen/experiments/r03.py)):

Un decoder XCT más ancho y profundo que se entrena sobre el `mu` congelado del VAE. Su objetivo es responder: *"¿Contiene `mu` suficiente información para reconstruir XCT con alta calidad, siendo el cuello de botella el decoder original y no el espacio latente?"*

Arquitectura: stem Conv3d → N bloques de upsampling (`ConvTranspose3d` + `GroupNorm` + `SiLU` + residual) → cabeza Conv3d 1×1. Los canales son el doble que el decoder original del VAE.

**Cómo entrenarlo** ([experiments/r03.py:359-448](../src/poregen/experiments/r03.py)):
```python
aux_decoder, history_df = train_auxiliary_decoder(
    model,             # VAE congelado
    train_loader,
    val_loader,
    device,
    lr=3e-4,
    max_epochs=8,
    patience=2,        # early stopping
)
# Optimiza: Charbonnier(sigmoid(aux_decoder(mu)), xct_gt)
```

**Razonamiento:** Si `xct_loss_auxiliary` << `xct_loss_original` con el mismo `mu`, el espacio latente es rico pero el decoder original del VAE es el cuello de botella. Si ambas son similares, el límite está en la información que `mu` codifica de la textura XCT — el VAE ha comprimido demasiado.

---

### `evaluate_auxiliary_decoder` — DataFrame de análisis por parche

**Fórmula** ([experiments/r03.py:452-528](../src/poregen/experiments/r03.py)):

Devuelve un DataFrame con **una fila por parche** y las siguientes columnas:

| Columna | Fórmula | Interpretación |
|---|---|---|
| `porosity` | `mask.mean(dim=(1,2,3,4))` | Fracción volumétrica de poros del parche |
| `transition_density` | `local_mask_variance_map(mask, kernel=5).mean()` | Densidad de transiciones poro/sólido — ve abajo |
| `transition_label` | `"high_transition"` si `transition_density >= percentil_75` | Clasifica el parche como interior (homogéneo) o de transición |
| `xct_loss_original` | `charbonnier(sigmoid(original_decoder(z)), xct_gt)` por parche | Error de reconstrucción del decoder original |
| `xct_loss_auxiliary` | `charbonnier(sigmoid(aux_decoder(mu)), xct_gt)` por parche | Error de reconstrucción del decoder auxiliar más potente |
| `sharpness_gt` | `samplewise_sharpness_proxy(xct_gt)` | Nitidez del GT por parche |
| `sharpness_recon_original` | `samplewise_sharpness_proxy(sigmoid(logits_original))` | Nitidez reconstrucción original |
| `sharpness_recon_auxiliary` | `samplewise_sharpness_proxy(sigmoid(logits_aux))` | Nitidez reconstrucción auxiliar |
| `sharpness_ratio_original` | `sharpness_recon_original / sharpness_gt` | Ratio de nitidez, original |
| `sharpness_ratio_auxiliary` | `sharpness_recon_aux / sharpness_gt` | Ratio de nitidez, auxiliar |

---

### `transition_density` — Densidad de transiciones poro/sólido

**Fórmula** ([experiments/r03.py:169-189](../src/poregen/experiments/r03.py)):

Dos proxies disponibles (equivalentes en interpretación, distintos en implementación):

**`mask_local_variance_density`** (kernel de promedio local):
```python
mean    = avg_pool3d(mask, kernel_size=5, stride=1, padding=2)
mean_sq = avg_pool3d(mask², kernel_size=5, stride=1, padding=2)
local_var = (mean_sq - mean²).clamp(0)          # (B,1,D,H,W) mapa de varianza local
transition_density = local_var.mean(dim=(1,2,3,4))  # (B,) escalar por parche
```

**`mask_gradient_density`** (diferencias finitas):
```python
gz = |mask[:,:,1:,:,:] - mask[:,:,:-1,:,:]|.mean()  # por cada eje
transition_density = (gz + gy + gx) / 3.0
```

**Razonamiento:** Los parches con alta densidad de transición son los más difíciles para el modelo — contienen muchos bordes poro/sólido. Los parches "interior" (baja transición) son mayoritariamente homogéneos (todo sólido o todo poro). Si `xct_loss_original` es alto principalmente en parches `high_transition`, la causa es la dificultad de reconstruir bordes, no falta de información latente.

**Interpretación:** Usar `transition_percentile_threshold(scores, percentile=75)` para determinar el umbral de clasificación. Analizar la distribución de `xct_loss_original` y `sharpness_ratio_original` separado por `transition_label`:
- Si `high_transition` tiene pérdidas mucho mayores → el problema es la reconstrucción en bordes.
- Si las pérdidas son similares en ambos grupos → el problema es global (capacidad del latente o del decoder).

---

### `samplewise_charbonnier` y `samplewise_sharpness_proxy`

Versiones por parche (retornan tensor `(B,)`) de las métricas globales del training loop:

**`samplewise_charbonnier`** ([experiments/r03.py:147-154](../src/poregen/experiments/r03.py)):
```python
diff = pred - target
samplewise_charbonnier[b] = sqrt(diff[b]² + eps²).mean()
```

**`samplewise_sharpness_proxy`** ([experiments/r03.py:157-162](../src/poregen/experiments/r03.py)):
```python
samplewise_sharpness[b] = (|grad_d[b]| + |grad_h[b]| + |grad_w[b]|).mean() / 3
```

**Razonamiento:** Las métricas globales del training loop agregan sobre el batch y pueden enmascarar que la mayoría del error viene de unos pocos parches difíciles. Las versiones sample-wise permiten distribuciones, scatter plots (ej. pérdida vs porosidad, pérdida vs transition_density) y detección de outliers.

---

### `encode_mu` — Codificación determinista para análisis

**Fórmula** ([experiments/r03.py:122-126](../src/poregen/experiments/r03.py)):
```python
x = cat([xct, mask], dim=1)
h = model.encoder(x)
mu = model.to_mu(h)     # sin muestreo: no hay eps
```

**Cuándo usarlo:** En análisis offline (PCA del espacio latente, clustering, pipeline de difusión). Usar `mu` en lugar de `z` elimina el ruido de reparametrización, dando representaciones más estables y comparables entre parches.

---

## 7. Evaluación final completa (val_full/ / test_full/)

Calculada una sola vez al final del entrenamiento si `final_full_eval=true` (R03). Pasa por **todos** los batches del loader (no un subconjunto fijo), loguea bajo prefijos `val_full/` y `test_full/`.

**Razonamiento:** Las evaluaciones periódicas usan un subconjunto (100 batches de val, 20 de test) para no bloquear el entrenamiento. La evaluación final usa el dataset completo para obtener métricas definitivas libres de ruido de submuestra.

**Cómo leerla:** En TensorBoard, filtrar por `val_full` o `test_full`. El único step registrado es el final. Estos valores son los que deben reportarse como resultado definitivo del experimento.

---

## 8. Imágenes en TensorBoard

Se logean cada `image_log_every=62` steps para val, y en cada evaluación de test, bajo `{val,test}/`:

| Tag | Contenido | Cómo generarlo |
|---|---|---|
| `{prefix}/xct_gt_{d,h,w}` | Corte central XCT ground truth | `xct_gt.clamp(0,1)` |
| `{prefix}/xct_recon_{d,h,w}` | Corte central XCT reconstruido | `xct_logits.clamp(0,1)` |
| `{prefix}/mask_gt_{d,h,w}` | Corte central máscara GT (poros en blanco) | `mask_gt` binario |
| `{prefix}/mask_recon_{d,h,w}` | Corte central máscara predicha | `sigmoid(mask_logits)` continuo |

Los ejes son d (profundidad/axial), h (alto/coronal), w (ancho/sagittal).

**Artefactos comunes:**
- **Borroneo**: recon suave sin bordes → `sharpness_recon_over_gt` < 1.0
- **Halo**: zona brillante/oscura alrededor de poros en la máscara → FP en borde de poros
- **Poros fantasma**: la máscara predicha tiene manchas donde no hay poros en GT → precision baja
- **Poros perdidos**: el GT tiene poros que la máscara no detecta → recall bajo
- **Checkerboard**: patrón de tablero ajedrez en xct_recon → artefacto de transposed convolutions, `sharpness_recon_over_gt` > 1.0

---

## 9. Archivos generados en disco

| Archivo | Contenido | Cuándo se crea |
|---|---|---|
| `log.jsonl` | Un registro JSON por step: todas las pérdidas train, activos latentes, elapsed. Los registros de val se añaden también aquí. | Cada step + cada `eval_every` |
| `metrics.jsonl` | Solo registros de val y test (subconjunto de `log.jsonl`). Usado para análisis offline. | Cada `eval_every` / `test_every` |
| `{name}_step{N:08d}.ckpt` | Checkpoint completo: `model_state_dict`, `optimizer_state_dict`, `scaler_state_dict`, `scheduler_state_dict`, `step`, `metadata`. | Cada `save_every=1000` steps y al final |
| `samples/step_{N:08d}/{split}/` | Arrays 3D exportados: `xct_gt`, `mask_gt`, `xct_recon`, `mask_recon` (N,1,64,64,64). | Cada `sample_every=12500` steps |
| `samples/step_{N:08d}/{split}_meta.json` | Metadatos por parche: `volume_id`, `z0`, `y0`, `x0`, `porosity`, `source_group`. | Ídem |

**Nota sobre `log.jsonl`:** Las listas (como `kl_per_channel`) se filtran del JSON para no saturar el archivo — solo se guardan escalares. Los vectores por canal solo están disponibles en TensorBoard.

---

## 10. Resumen de prioridades

| Criterio | Métrica | Target |
|---|---|---|
| **Éxito del proyecto** | `porosity_mae` | < 0.005 |
| **Calidad visual XCT** | `psnr` | > 30 dB |
| **Nitidez XCT** | `sharpness_recon_over_gt` | ≈ 1.0 (rango sano: 0.9–1.1) |
| **Segmentación** | `dice_pos_only` | > 0.5 prometedor; > 0.7 bueno |
| **Espacio latente sano (μ-varianza)** | `mu_active_fraction` (val) | ≈ 1.0 |
| **Espacio latente sano (KL)** | `kl_active_channels` (train) | = C (todos activos) |
| **Sin colapso KL** | `kl_raw` | > `free_bits × C` = 2.0 |
| **Sin colapso determinista** | `xct_diversity` (MC) | > 0.01 |
| **Sesgo de porosidad** | `porosity_bias` | ≈ 0.0 |
| **Robustez entre volúmenes** | `porosity_mae_vol_p90` | < 0.01 |
| **Estabilidad de entrenamiento** | `grad_norm` | < 2.0 (sin picos sostenidos) |
| **Regularización KL efectiva** | `kl_collapsed_fraction` | → 0.0 al final |

### Señales de alerta rápida

| Síntoma en TensorBoard | Diagnóstico probable |
|---|---|
| `kl` ≈ `free_bits × C` de forma persistente | Posterior collapse: todos los canales clamped |
| `kl_collapsed_fraction` ≈ 1.0 después de muchos steps | Idem; considera reducir `kl_max_beta` o aumentar `kl_warmup_steps` |
| `kl_active_channels` (train) estancado en < C | Colapso KL parcial: algunos canales permanecen clamped |
| `mu_active_fraction` (val) estancada en < 0.5 | Colapso parcial μ-varianza: el encoder asigna μ casi constante |
| `xct_diversity` → 0 o < 1e-5 | El VAE actúa como autoencoder determinista; la std σ ha colapsado |
| `sharpness_recon_over_gt` << 1.0 y no sube | Borroneo crónico; el decoder promedia posibilidades en lugar de comprometerse |
| `mask_pred_mean` → 0 | Colapso de la cabeza de máscara: nunca predice poros |
| `porosity_bias` grande con `porosity_mae` bajo | Sesgo sistemático que se cancela en el promedio; verificar distribución por volumen |
| `porosity_mae_bin_3` >> `porosity_mae_bin_0..2` | El modelo falla en alta porosidad; revisar equilibrio del dataset o pos_weight |
| `test/porosity_mae_per_volume` con cola larga / `vol_p90` alto | Subgrupo de volúmenes no cubierto bien; revisar splits y distribución de porosidades |
| `grad_scaler_scale` cayendo repetidamente | Overflow de gradientes frecuente; reducir lr o max_grad_norm |
| `grad_norm_encoder` >> `grad_norm_decoder` | Gradientes explosivos localizados en encoder; revisar arquitectura o lr |
