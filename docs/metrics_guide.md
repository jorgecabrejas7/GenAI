# Guía de Métricas — Pipeline de Entrenamiento VAE

---

## 1. LOSSES

Las pérdidas se calculan en cada step de train y de eval. La función principal es `compute_total_loss` en [losses/total.py](../src/poregen/losses/total.py).

---

### `total`

**Fórmula:**
```
total = xct_weight × xct_loss + mask_total + beta × kl
```

**Cómo se calcula:** Es la suma ponderada de los tres componentes principales: reconstrucción XCT, pérdida de máscara, y divergencia KL. Es la única pérdida que se propaga hacia atrás con `.backward()`.

**Cómo interpretarla:** Debe bajar de forma estable a lo largo del entrenamiento. Un descenso suave en train con un descenso paralelo en val indica que el modelo generaliza bien. Si val sube mientras train baja, hay overfitting. Oscilaciones fuertes suelen indicar lr demasiado alto o batch size inadecuado.

---

### `xct_loss` — Reconstrucción XCT (Charbonnier)

**Fórmula:**
```python
diff = pred_logits - target
xct_loss = mean( sqrt(diff² + 1e-12) )
```

**Cómo se calcula:** Se aplica directamente entre los logits del decoder (sin activación, en espacio z-score) y el target XCT normalizado en [0, 1]. La variante Charbonnier es una versión suavizada del L1: cuando el error es grande se comporta como L1 (gradiente constante), y cerca de cero es diferenciable, a diferencia de L1 puro que tiene gradiente discontinuo en 0.

**Cómo interpretarla:** Mide el error de reconstrucción vóxel a vóxel. Valores < 0.01 son buenos. Un valor que no baja de ~0.05 indica que el decoder tiene dificultades para recuperar la textura XCT. Si baja muy rápido mientras el KL no sube, puede que la red esté memorizando en lugar de aprender una representación latente útil.

---

### `mask_bce` — Binary Cross-Entropy con pos_weight

**Fórmula:**
```python
bce = -[ pos_weight × y × log(σ(logit)) + (1-y) × log(1-σ(logit)) ]
# con pos_weight = config["mask_bce_pos_weight"]  (en config: 17.16)
```

**Cómo se calcula:** Se usa `F.binary_cross_entropy_with_logits` que es numéricamente estable porque combina sigmoid y BCE internamente. El `pos_weight` multiplica la pérdida en los vóxeles de poro (y=1), compensando el desbalanceo de clases. Si la porosidad media es φ ≈ 0.055, un `pos_weight ≈ (1-φ)/φ ≈ 17` hace que un falso negativo (poro no detectado) cueste ~17× más que un falso positivo.

**Cómo interpretarla:** Sin `pos_weight`, la red aprendería a predecir siempre cero y acertaría el 94.5% de los vóxeles trivialmente. Con él, la pérdida es aproximadamente igual de sensible a ambas clases. Un valor que baja pero luego se estanca puede indicar que la red aprende la frecuencia global de los poros pero no su forma o posición exacta.

---

### `mask_tversky` — Pérdida de Tversky

**Fórmula:**
```python
pred = sigmoid(logits)
tp = (pred × target).sum(dim=1)          # por muestra, aplanado espacialmente
fp = (pred × (1 - target)).sum(dim=1)
fn = ((1 - pred) × target).sum(dim=1)

tversky_coeff = (tp + 1) / (tp + 0.3×fp + 0.7×fn + 1)
mask_tversky = 1 - mean(tversky_coeff)
```

**Cómo se calcula:** Es una generalización del coeficiente Dice donde FP y FN tienen pesos distintos. Con α=0.3 y β=0.7, un falso negativo (poro no detectado) penaliza 2.3× más que un falso positivo (poro fantasma). Esto prioriza el recall sobre la precisión, lo cual es apropiado para poros pequeños y dispersos donde perderlos es más costoso que predecir alguno extra.

**Cómo interpretarla:** Rango [0, 1]. Un valor de 0 indica segmentación perfecta. Un valor cercano a 1 indica que la red no detecta poros en absoluto. Si baja más despacio que `mask_bce`, indica que la forma y localización exacta de los poros (lo que captura Tversky) es más difícil de aprender que su frecuencia global (lo que captura BCE).

---

### `mask_total`

**Fórmula:**
```python
mask_total = bce_weight × mask_bce + dice_weight × mask_tversky
# con bce_weight=1.0 y dice_weight=1.0 en config
```

**Cómo interpretarla:** Suma de ambas pérdidas de máscara. BCE aporta señal de gradiente vóxel a vóxel; Tversky aporta señal a nivel de región. Usar ambas es más robusto que cualquiera por separado: BCE sola converge a predecir la frecuencia base sin forma; Tversky sola tiene gradientes ruidosos al inicio cuando la red predice todo cero.

---

### `kl` — Divergencia KL

**Fórmula:**
```python
# Por elemento: (B, C, d, h, w)
kl_elem = 0.5 × (μ² + exp(logvar) - logvar - 1)

# Media sobre batch y dimensiones espaciales → (C,) por canal
kl_per_channel = kl_elem.mean(dim=(0, 2, 3, 4))

# Free-bits: clamp mínimo por canal
kl_per_channel_clamped = max(kl_per_channel, free_bits=0.25)

# Suma escalar
kl = kl_per_channel_clamped.sum()
```

**Cómo se calcula:** Es la KL analítica entre la distribución posterior del encoder `q(z|x) = N(μ, σ²)` y la prior `p(z) = N(0, I)`. El free-bits de 0.25 garantiza que cada canal tenga al menos ese nivel de KL antes de contribuir al total, lo que evita el posterior collapse (canales que colapsan a la prior y son ignorados por el decoder).

**Cómo interpretarla:** Si `kl` es 0 o muy próximo a 0, el encoder está enviando la prior directamente y el espacio latente es inútil (posterior collapse). Si es muy alto, el espacio latente no está regularizado y la generación aleatoria producirá ruido. El rango sano depende del número de canales y del free_bits: con C=8 canales y free_bits=0.25, el mínimo esperado es 8 × 0.25 = 2.0. Valores entre 2 y 10 son razonables según el beta usado.

---

### `beta` — Peso dinámico del KL

**Fórmula:**
```python
beta = min(step / kl_warmup_steps, 1.0) × kl_max_beta
# kl_warmup_steps=28676, kl_max_beta=0.05
```

**Cómo se calcula:** Rampa lineal de 0 a 0.05 durante las primeras 2 épocas (~28k steps). Después permanece fijo en 0.05.

**Cómo interpretarla:** No es una métrica de calidad sino un hiperparámetro en escala temporal. Su curva en TensorBoard debe ser una rampa perfectamente lineal hasta el plateau. Si ves que el KL colapsa antes de que beta llegue a su valor final, considera alargar `kl_warmup_steps`.

---

### `freebits_used`

**Fórmula:**
```python
n_clamped = (kl_per_channel < free_bits).float().mean()
# = fracción de canales cuyo KL raw < 0.25
```

**Cómo interpretarla:** Fracción de canales que están por debajo del umbral free_bits y por tanto están siendo "subvencionados" (su KL real es inferior al mínimo garantizado). Al inicio del entrenamiento, todos los canales estarán clamped (freebits_used ≈ 1.0). Con el tiempo, a medida que los canales aprenden a codificar información, su KL sube y salen del clamp. Al final, `freebits_used` próximo a 0 indica que todos los canales están activos y contribuyen con KL propio.

---

## 2. MÉTRICAS DE TRAIN

Se logean en TensorBoard bajo el prefijo `train/` cada `log_every` steps (por defecto cada step).

---

### `train/grad_norm`

**Fórmula:**
```python
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm=1.0)
```

**Cómo se calcula:** Es la norma L2 global del gradiente **antes** de aplicar el clipping. Se calcula después de `scaler.unscale_()` para que sea comparable independientemente del GradScaler.

**Cómo interpretarla:** Si consistentemente vale exactamente 1.0, el clipping está activo en cada step y el lr puede ser demasiado alto. Valores entre 0.1 y 0.8 indican gradientes sanos. Picos ocasionales > 1.0 son normales (especialmente al inicio). Un grad_norm que crece de forma sostenida puede indicar inestabilidad numérica o un lr que hay que reducir.

---

### `train/lr`

**Cómo se calcula:** Learning rate actual tomado del scheduler cosine con warmup. Sube linealmente de 0 a 4e-4 durante `warmup_steps=7169` y luego decae en coseno hasta `lr_min=4e-5`.

**Cómo interpretarla:** Confirma que el scheduler está funcionando correctamente. La curva debe mostrar una subida lineal corta seguida de un descenso suave en coseno. Útil para correlacionar cambios en el loss con cambios en lr.

---

### `train/kl_ch00` ... `train/kl_ch07`

**Cómo se calcula:** KL raw (pre-clamp) de cada uno de los C=8 canales latentes, calculado como `kl_per_channel[i]`.

**Cómo interpretarla:** Permite detectar qué canales están activos y cuáles están colapsados. Un canal activo tiene KL creciente que supera el free_bits. Un canal muerto se queda atascado en KL ≈ 0 o ≈ free_bits indefinidamente. Si todos los canales muestran la misma KL y no crecen, hay colapso total.

---

### `train/active_channels`

**Fórmula:**
```python
n_dead = round(freebits_used × len(kl_chs))
active_channels = len(kl_chs) - n_dead
```

**Cómo interpretarla:** Número de canales latentes cuya KL supera el free_bits. Idealmente debe crecer desde 0 al inicio del entrenamiento hasta C=8 (todos activos). Si se estanca en un valor bajo (p.ej. 2 de 8), los canales restantes están colapsados y el modelo está infrautilizando su capacidad latente.

---

## 3. MÉTRICAS DE VALIDACIÓN

Se calculan cada `eval_every=62` steps, promediando sobre `val_batches=20` batches. Prefijo `val/` en TensorBoard.

---

### `val/mae` — Mean Absolute Error XCT

**Fórmula:**
```python
mae = F.l1_loss(output.xct_logits, xct_target)
    = mean( |xct_logits - xct_gt| )
```

**Cómo se calcula:** Error absoluto medio entre la predicción del decoder y el volumen XCT de referencia. Ambos están en [0, 1] (normalización uint8/255). A diferencia de la loss de entrenamiento (Charbonnier en z-score), esta es una métrica de evaluación pura sin ponderaciones.

**Cómo interpretarla:** Rango [0, 1]. Un MAE de 0.01 significa que los vóxeles reconstruidos difieren en promedio un 1% del rango de intensidad. Valores < 0.02 son buenos; < 0.01 son excelentes. Si MAE es bajo pero las imágenes visuales parecen borronas, hay que mirar también sharpness.

---

### `val/psnr` — Peak Signal-to-Noise Ratio

**Fórmula:**
```python
mse = F.mse_loss(xct_logits, xct_gt)
psnr = 10 × log₁₀(1.0² / mse)   # max_val=1.0
```

**Cómo se calcula:** Relación señal-ruido de pico en decibelios, usando MSE como medida de error. El valor máximo (1.0²) es 1 porque ambas señales están en [0, 1].

**Cómo interpretarla:** Escala logarítmica: cada +3 dB significa que el error se reduce a la mitad. Valores de referencia: > 30 dB = buena calidad; > 35 dB = excelente. PSNR alto no garantiza buena percepción visual (puede haber borroneo uniforme con PSNR aceptable), por eso se complementa con `sharpness`.

---

### `val/sharpness_recon` y `val/sharpness_gt`

**Fórmula:**
```python
gd = (x[:, :, 1:, :, :] - x[:, :, :-1, :, :]).abs().mean()  # gradiente en D
gh = (x[:, :, :, 1:, :] - x[:, :, :, :-1, :]).abs().mean()  # gradiente en H
gw = (x[:, :, :, :, 1:] - x[:, :, :, :, :-1]).abs().mean()  # gradiente en W
sharpness = (gd + gh + gw) / 3.0
```

**Cómo se calcula:** Proxy de nitidez basado en la magnitud media del gradiente por diferencias finitas en los tres ejes espaciales. Se calcula tanto para la reconstrucción (`sharpness_recon`) como para el ground truth (`sharpness_gt`).

**Cómo interpretarla:** `sharpness_gt` es la referencia fija del dataset. `sharpness_recon` debe converger hacia ese valor. Si `sharpness_recon` << `sharpness_gt`, la red produce reconstrucciones borronas (frecuencia espacial baja). Si `sharpness_recon` > `sharpness_gt`, la red está introduciendo artefactos de alta frecuencia (ruido). La ratio `sharpness_recon / sharpness_gt` es el indicador más directo: idealmente cercana a 1.0.

---

### `val/recon_xct_mean`

**Fórmula:**
```python
recon_xct_mean = output.xct_logits.mean()
```

**Cómo interpretarla:** Media global de los vóxeles reconstruidos. Debe estar próxima a la media del GT (~0.5 en escala normalizada). Una desviación grande indica un sesgo sistemático: el decoder predice intensidades sistemáticamente más altas o más bajas que las reales.

---

### `val/recon_xct_std`

**Fórmula:**
```python
recon_xct_std = output.xct_logits.std()
```

**Cómo interpretarla:** Desviación estándar de los vóxeles reconstruidos. Una std muy baja (mucho menor que la del GT) indica predicciones "planas" o borronas donde el decoder promedia los posibles valores en lugar de comprometerse con una textura concreta. Es una señal de que el espacio latente no tiene suficiente información para reconstruir los detalles.

---

### `val/dice_all` y `val/dice_pos_only`

**Fórmula:**
```python
# Por muestra b:
tp = (pred[b] × target[b]).sum()
fp = (pred[b] × (1 - target[b])).sum()
fn = ((1 - pred[b]) × target[b]).sum()
dice[b] = (2×tp + 1e-7) / (2×tp + fp + fn + 1e-7)

# dice_all: media sobre todos los b
# dice_pos_only: media sobre los b donde target[b].sum() > 0
```
con `pred = (sigmoid(logits) >= 0.5).float()`

**Cómo se calcula:** Coeficiente Dice medio, calculado por muestra (patch) y luego promediado. La variante `_pos_only` excluye los parches sin ningún poro en el GT.

**Cómo interpretarla:** `dice_all` está inflado artificialmente porque incluye parches vacíos donde tanto pred como target son todo cero, dando Dice=1 trivialmente. **`dice_pos_only` es la métrica relevante**: mide la capacidad de la red de segmentar poros cuando realmente los hay. Dice > 0.5 es prometedor; > 0.7 es bueno para poros pequeños y dispersos.

---

### `val/iou_all` y `val/iou_pos_only`

**Fórmula:**
```python
iou[b] = (tp + 1e-7) / (tp + fp + fn + 1e-7)
```

**Cómo interpretarla:** Más estricto que Dice porque el denominador es mayor (IoU = Dice/(2-Dice)). Dice=0.7 corresponde a IoU≈0.54. Úsalo como métrica complementaria, pero Dice es más habitual en segmentación médica. La distinción `_all` / `_pos_only` aplica igual que en Dice.

---

### `val/precision_all` y `val/precision_pos_only`

**Fórmula:**
```python
# Agregado sobre todos los b del batch
precision = (TP_total + 1e-7) / (TP_total + FP_total + 1e-7)
```

**Cómo interpretarla:** "De todos los vóxeles que la red marca como poro, ¿qué fracción son realmente poro?" Una precision baja indica muchos falsos positivos: la red detecta poros donde no los hay. Una precision alta con recall bajo significa que la red es conservadora: cuando dice "poro", casi siempre acierta, pero se pierde muchos.

---

### `val/recall_all` y `val/recall_pos_only`

**Fórmula:**
```python
recall = (TP_total + 1e-7) / (TP_total + FN_total + 1e-7)
```

**Cómo interpretarla:** "De todos los poros reales, ¿qué fracción detecta la red?" Un recall bajo significa que la red pierde muchos poros (FN altos). Dado que la pérdida Tversky penaliza los FN más que los FP (β=0.7 > α=0.3), el entrenamiento está sesgado hacia mejorar el recall.

---

### `val/f1_all` y `val/f1_pos_only`

**Fórmula:**
```python
f1 = 2 × precision × recall / (precision + recall + 1e-7)
```

**Cómo interpretarla:** Media harmónica de precision y recall. Para el caso binario es matemáticamente equivalente al Dice por muestra. Aquí se calcula de forma distinta (sobre TP/FP/FN acumulados del batch) por lo que puede diferir ligeramente de `dice`. Úsalo para confirmar consistencia entre las dos formas de calcular la misma cosa.

---

### `val/porosity_mae` — Error de Porosidad (Métrica Principal)

**Fórmula:**
```python
pred_por = sigmoid(mask_logits).mean(dim=(1,2,3,4))  # (B,) porosidad predicha por parche
gt_por   = mask_target.mean(dim=(1,2,3,4))            # (B,) porosidad real por parche
signed   = pred_por - gt_por
porosity_mae = signed.abs().mean()
```

**Cómo se calcula:** Para cada parche del batch, se calcula la fracción de volumen predicha como poro (promedio del sigmoid) y se compara con la fracción real. El MAE es el error absoluto medio de esa comparación.

**Cómo interpretarla:** Esta es la **métrica de éxito primaria del proyecto**. Target: < 0.005 (equivalente al 0.5% de volumen poroso, que es un cuarto de la porosidad media φ_mean ≈ 0.02). Un error de 0.005 significa que si un parche tiene 2% de poros, la red puede predecir entre 1.5% y 2.5%. Nota importante: esta métrica puede ser baja incluso si la segmentación espacial es imperfecta — la red puede acertar la cantidad total de poros pero tenerlos en sitios equivocados.

---

### `val/porosity_bias`

**Fórmula:**
```python
porosity_bias = (pred_por - gt_por).mean()
```

**Cómo interpretarla:** Sesgo sistemático en la estimación de porosidad. Un valor positivo significa que la red sobreestima la porosidad (predice más poros de los que hay). Un valor negativo significa que la subestima. Un bias próximo a 0 con MAE bajo es lo ideal. Si MAE es bajo pero bias es alto, el modelo tiene un sesgo sistemático que se cancela en el promedio — señal de que algo está mal en las predicciones individuales.

---

### `val/mask_pred_mean`

**Fórmula:**
```python
mask_pred_mean = sigmoid(mask_logits).mean()
```

**Cómo interpretarla:** Media global de las probabilidades de poro predichas por la red. Debe converger a ≈ φ_mean (≈ 0.055 según el subagent, aunque el target de porosity_mae sugiere φ_mean ≈ 0.02). Si este valor está cercano a 0, la red no está prediciendo poros (colapso de la máscara). Si está cercano a 0.5, la red no está discriminando. Es un indicador rápido de salud de la cabeza de segmentación.

---

### `val/active_fraction`

**Fórmula:**
```python
# Flatten mu a (C, N) donde N = B × d × h × w
mu_flat = mu.permute(1,0,2,3,4).reshape(C, -1)  # (C, N)
var = mu_flat.var(dim=1)                          # (C,) varianza por canal
n_active = (var > 0.01).sum()
active_fraction = n_active / C
```

**Cómo interpretarla:** Fracción de canales del espacio latente que codifican información real (varianza de μ > 0.01 sobre el batch). Si `active_fraction` = 1.0, todos los canales están siendo utilizados por el encoder. Si es baja (< 0.5), hay colapso parcial: el encoder está comprimiendo la información en pocos canales y el resto son ruido. Debe crecer durante el entrenamiento y estabilizarse cerca de 1.0.

---

### `val/n_active` y `val/n_total`

**Cómo interpretarlos:** Equivalente absoluto de `active_fraction`. `n_total` siempre será C=8 (número de canales latentes). `n_active` debe tender a 8 con el entrenamiento. Si se queda en 2-3, la red ha colapsado parcialmente.

---

### `val/mu_mean`

**Fórmula:**
```python
mu_mean = mu.mean()
```

**Cómo interpretarla:** Media global de los valores μ del encoder. La prior es N(0, I), así que μ debería tener media ≈ 0 si la regularización KL es efectiva. Un valor muy alejado de 0 indica que el encoder está usando un offset sistemático, lo que puede causar problemas si se muestrea de la prior durante generación.

---

### `val/mu_std`

**Fórmula:**
```python
mu_std = mu.std()
```

**Cómo interpretarla:** Dispersión de los valores μ. Si es ≈ 0, todos los parches producen el mismo código latente (sin información). Si es muy alta, el espacio latente está mal regularizado. Valores entre 0.5 y 2.0 suelen indicar un encoder activo y regularizado.

---

### `val/logvar_mean`

**Fórmula:**
```python
logvar_mean = logvar.mean()
```

**Cómo interpretarla:** Media del logaritmo de la varianza posterior. Un valor de 0 corresponde a σ=1, que coincide con la prior. Valores muy negativos (ej. -5) significan σ ≈ 0.08: la red está muy segura de sus códigos latentes pero se aleja de la prior (colapso hacia una distribución puntual). Valores positivos indican alta incertidumbre. El balance sano es cercano a 0.

---

### `val/logvar_std`

**Fórmula:**
```python
logvar_std = logvar.std()
```

**Cómo interpretarla:** Dispersión del logvar. Una std alta indica que el encoder asigna distintos niveles de certeza a distintos parches o regiones espaciales, lo cual es deseable: zonas homogéneas deberían tener mayor certeza que zonas complejas. Una std baja indica que el encoder trata todos los parches con la misma incertidumbre, sin distinción.

---

### `val/std_mean`

**Fórmula:**
```python
std_mean = exp(0.5 × logvar).mean()
# = mean(σ)
```

**Cómo interpretarla:** Media de la desviación estándar posterior σ. Al inicio del entrenamiento (antes del warmup KL), debe ser ≈ 1.0 (igual que la prior). Con el entrenamiento puede bajar moderadamente (el encoder se vuelve más determinista), pero no debería colapsar a 0. Si `std_mean` < 0.1, la red está operando casi como un autoencoder determinista y la regularización KL es insuficiente.

---

### `val/kl_total`

**Fórmula:**
```python
kl_total = sum(kl_per_channel)   # (sin el clamp de free_bits)
```

**Cómo interpretarla:** Suma de KL raw por canal (antes del free-bits clamping). Útil para detectar colapso: si `kl_total` ≈ 0, ningún canal está codificando información. Compara con `kl` (que usa el clamp): si `kl_total` << `kl`, significa que muchos canales son inactivos y están siendo subvencionados por el free-bits.

---

## 4. MÉTRICAS DE TEST

Se calculan cada `test_every=625` steps, promediando sobre `test_batches=20` batches. Prefijo `test/` en TensorBoard. Todas las métricas de validación se replican aquí exactamente con las mismas fórmulas e interpretaciones. La única métrica adicional exclusiva de test es:

---

### `test/porosity_mae_per_volume` — Histograma de error por volumen

**Fórmula:**
```python
# Durante eval, acumular por volume_id:
vol_por_errors[volume_id].append( (pred_por_patch - gt_por_patch).item() )

# Al final, por volumen:
volume_mae = abs( mean(signed_errors_for_this_volume) )
# → histograma de estos valores
```

**Cómo se calcula:** Para cada parche se acumula el error signado de porosidad (pred - gt). Al final del eval, por cada volumen se toma la media de esos errores signados (estimación del sesgo sistemático por volumen) y se toma su valor absoluto. El resultado es un histograma donde cada barra representa un volumen del test set.

**Cómo interpretarla:** Permite detectar si hay **volúmenes concretos** donde la red falla sistemáticamente, cosa que el MAE global no revelaría (podría estar enmascarado por otros volúmenes con error bajo). Un histograma ideal tiene casi todas las barras cerca de 0 con una cola corta. Si aparece un pico secundario a la derecha (ej. en 0.02), hay un subgrupo de volúmenes con porosidades sistemáticamente mal predichas, lo que puede indicar un tipo de muestra no visto en train o un grupo específico del dataset que requiere atención.

---

## 5. OTRAS

### Imágenes en TensorBoard

Se logean cada `image_log_every=62` steps para val, y en cada evaluación de test.

```
{val,test}/xct_gt_{d,h,w}      — corte central del XCT original en cada eje
{val,test}/xct_recon_{d,h,w}   — corte central del XCT reconstruido
{val,test}/mask_gt_{d,h,w}     — máscara de poros original (poros en blanco)
{val,test}/mask_recon_{d,h,w}  — máscara predicha (sigmoid de logits)
```

Los logits XCT se clampean a [0, 1] para visualización. La máscara se pasa por sigmoid para mostrar probabilidades continuas (no binarizadas).

**Cómo interpretarlas:** Comparar GT vs recon en los tres ejes. Los artefactos más comunes son:
- **Borroneo**: recon suave sin bordes nítidos → `sharpness_recon` << `sharpness_gt`
- **Halo**: zona brillante o oscura alrededor de poros en la máscara → FP en borde de poros
- **Poros fantasma**: la máscara predicha tiene manchas donde no hay poros en el GT → precision baja
- **Poros perdidos**: el GT tiene poros que la máscara predicha no detecta → recall bajo

---

### Archivos generados

| Archivo | Contenido | Cuándo se crea |
|---|---|---|
| `log.jsonl` | Un registro JSON por step con todas las pérdidas train y val | Cada step |
| `metrics.jsonl` | Solo registros de val y test | Cada eval_every / test_every |
| `last.ckpt` | Checkpoint completo: modelo, optimizer, scaler, scheduler, step | Cada save_every=2500 steps |
| `samples/step_XXXXXXXX/{split}.npz` | Arrays 3D: xct_gt, mask_gt, xct_recon, mask_recon (N,1,64,64,64) | Cada sample_every=12500 steps |
| `samples/step_XXXXXXXX/{split}_meta.json` | Metadatos por parche: volume_id, coords, porosity, source_group | Ídem |

---

## Resumen de prioridades

| Criterio | Métrica | Target |
|---|---|---|
| **Éxito del proyecto** | `porosity_mae` | < 0.005 |
| **Calidad visual XCT** | `psnr` | > 30 dB |
| **Nitidez XCT** | `sharpness_recon / sharpness_gt` | ≈ 1.0 |
| **Segmentación** | `dice_pos_only` | > 0.5 prometedor, > 0.7 bueno |
| **Espacio latente sano** | `active_fraction` | ≈ 1.0 |
| **Sin colapso KL** | `kl_total` | > `free_bits × n_channels` = 2.0 |
| **Entrenamiento estable** | `grad_norm` | < 1.0 (sin clipping constante) |
