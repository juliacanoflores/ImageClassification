# Reporte de Experimentación

## 1) Model selection rationale

### Objetivo experimental
Entrenar y comparar modelos preentrenados de clasificación de imágenes para maximizar el rendimiento en clasificación multiclase de escenas sobre el conjunto de validación.

El objetivo no se limita a obtener la mayor exactitud posible, sino a identificar una solución robusta, reproducible y viable en coste computacional. Por este motivo, la comparación entre modelos debe realizarse en condiciones homogéneas (mismos datos, mismo criterio de parada y métricas comunes).

### Métrica principal
Usar `val_accuracy` como métrica principal, definida como la proporción de predicciones correctas en validación:

$$
val\_accuracy = \frac{\text{aciertos}}{\text{total de muestras}}
$$

Como métricas secundarias de apoyo, usar `val_loss` para evaluar calibración y estabilidad del entrenamiento, y considerar métricas por clase cuando existan diferencias claras entre categorías.

### Justificación de la métrica
- El problema es de clasificación multiclase.
- La métrica es fácil de interpretar y comparar entre ejecuciones.
- La métrica permite seleccionar rápidamente la mejor configuración durante el ajuste de hiperparámetros.

Para evitar conclusiones incompletas, la decisión final no debe basarse en un único punto de la curva ni en una sola corrida aislada.

### Criterio de éxito
Considerar exitoso el experimento si se alcanza al menos `val_accuracy >= 0.80` en validación, manteniendo curvas estables de entrenamiento y validación (sin sobreajuste severo).

Se considera sobreajuste severo cuando la brecha entre métricas de entrenamiento y validación crece de forma sostenida durante las últimas épocas, sin mejora real en validación.

### Criterio de selección del modelo final
1. Seleccionar el modelo con mayor `val_accuracy` en validación.
2. Si hay empate, priorizar menor `val_loss`.
3. Si el empate persiste, priorizar menor coste computacional (tiempo por época / memoria).

Arquitecturas candidatas recomendadas para justificar la elección:
- ResNet50 como baseline sólido y eficiente.
- EfficientNet-B0 por buena relación precisión/coste.
- ConvNeXt-Base cuando se prioriza rendimiento máximo y se dispone de más recursos.

Evidencia a incluir en el reporte:
- Tabla comparativa con mejor `val_accuracy`, mejor `val_loss`, tiempo medio por época y tamaño aproximado del modelo.
- Breve comentario técnico que explique por qué el modelo ganador supera a los demás en el contexto del dataset.

### Evidencia experimental (export W&B)

| Modelo | Epochs | Val accuracy | Val loss | Train accuracy | Train loss | Runtime total | Tiempo por época |
|---|---:|---:|---:|---:|---:|---|---|
| ConvNeXt-Base | 8 | 0.9453 | 0.1603 | 0.9069 | 0.2840 | 3h 44m 31s | 28m 04s |
| EfficientNet-B0 | 5 | 0.8993 | 0.3595 | 0.7946 | 0.6793 | 24m 46s | 4m 57s |
| ResNet50 | 1 | 0.4073 | 2.4012 | 0.2271 | 2.5494 | 5m 44s | 5m 44s |

### Ranking por criterio de selección
1. ConvNeXt-Base: mejor val_accuracy y menor val_loss.
2. EfficientNet-B0: segundo mejor rendimiento con menor coste computacional.
3. ResNet50: rendimiento claramente insuficiente con la configuración actual.

### Decisión de selección de modelo
Seleccionar ConvNeXt-Base como modelo final del experimento por dominar la métrica principal (val_accuracy) y la métrica de desempate (val_loss).

## 2) Transfer-learning strategy
La estrategia de transferencia debe seguir un esquema progresivo para preservar el conocimiento preentrenado y adaptarlo al dominio de escenas.

### Justificación del parámetro unfreezed_layers
El parámetro `unfreezed_layers` controla cuántos bloques finales del extractor de características se dejan entrenables. En términos prácticos:
- `unfreezed_layers = 0`: todo el backbone queda congelado y solo se entrena la cabeza de clasificación.
- `unfreezed_layers > 0`: se permite ajustar los últimos bloques del backbone para adaptar mejor las representaciones al dataset.

Para la primera fase del experimento, se fija `unfreezed_layers = 0` en todos los modelos. Esta decisión se justifica por tres motivos:
- Asegura una comparación inicial justa entre arquitecturas al reducir diferencias de protocolo.
- Reduce riesgo de sobreajuste temprano cuando la cabeza aún no está adaptada al problema.
- Disminuye coste computacional y estabiliza la convergencia inicial.

Una vez estabilizada la cabeza, se pasa a una segunda fase con `unfreezed_layers > 0` de forma controlada y con learning rate menor. Este esquema permite aprovechar el preentrenamiento sin degradar las características generales aprendidas en el backbone.

Fase 1: entrenamiento de la cabeza
- Congelar el backbone completo.
- Entrenar únicamente la cabeza de clasificación durante pocas épocas para adaptar la salida al número de clases.

Fase 2: fine-tuning parcial
- Descongelar las últimas capas o bloques del backbone.
- Reducir el learning rate respecto a la fase 1 para evitar destruir características útiles aprendidas en preentrenamiento.

Buenas prácticas adicionales
- Mantener transformaciones coherentes con el tamaño de entrada del modelo.
- Usar regularización (por ejemplo, weight decay) para limitar sobreajuste.
- Guardar el mejor checkpoint según validación y no solo el último estado entrenado.
- Si es posible, usar early stopping cuando no haya mejoras tras varias épocas.

## 3) Hyperparameter tuning depth
La calidad del ajuste de hiperparámetros depende de la cobertura del espacio de búsqueda y de la consistencia del protocolo experimental.

Espacio mínimo recomendado
- Learning rate: al menos 3 o 4 valores en escala logarítmica.
- Optimizador: comparar al menos dos enfoques (por ejemplo, AdamW frente a SGD).
- Weight decay: evaluar varios niveles de regularización.
- Batch size: probar al menos dos opciones si el hardware lo permite.
- Capas descongeladas: contrastar diferentes profundidades de fine-tuning.

Profundidad esperada
- Ejecutar al menos 10 corridas comparables; objetivo recomendado entre 12 y 20.
- Mantener constantes las variables no estudiadas para atribuir efectos de manera fiable.
- Repetir configuraciones prometedoras para comprobar estabilidad.

Cómo presentar resultados
- Incluir ranking de configuraciones por métrica principal.
- Explicar patrones observados (por ejemplo, learning rate alto acelera convergencia pero degrada validación).
- Justificar por qué la configuración final es un compromiso adecuado entre precisión y robustez.

## 4) Calidad de tracking en W&B
El seguimiento en W&B debe permitir reconstruir cualquier decisión experimental sin ambiguedades.

Estandarización de runs
- Definir una convención de nombres y tags (modelo, fase, versión de experimento, fecha o índice).
- Agrupar ejecuciones por familia de pruebas para facilitar comparaciones.

Información mínima a registrar
- Configuración completa por corrida: modelo, learning rate, optimizer, weight decay, batch size, épocas, seed y política de capas descongeladas.
- Métricas por época: `train_loss`, `val_loss`, `val_accuracy`.
- Señales de proceso: learning rate efectivo, tiempo por época y duración total.

Artefactos y visualización
- Subir pesos del mejor modelo como artefacto versionado.
- Registrar gráficas de entrenamiento/validación y, si es posible, matriz de confusión.
- Mantener un resumen final por run con el mejor valor de cada métrica clave.

## 5) Reproducibilidad
La reproducibilidad es un criterio central de calidad experimental, no un extra opcional.

Requisitos técnicos
- Fijar semillas en Python, NumPy y PyTorch (incluyendo configuración de CUDA cuando aplique).
- Registrar versión de librerías y del entorno de ejecución.
- Documentar hardware usado (tipo de GPU/CPU y memoria relevante).

Requisitos metodológicos
- Guardar la configuración exacta del mejor run.
- Repetir la mejor configuración para comprobar que el resultado no depende de una corrida afortunada.
- Definir una tolerancia razonable de variación entre repeticiones y reportarla.

## 6) Análisis final
El análisis final debe conectar evidencia cuantitativa, comportamiento de entrenamiento y decisión de modelo.

### Interpretación de resultados
- ConvNeXt-Base logra el mejor compromiso en calidad predictiva, con una diferencia de +0.0460 en val_accuracy frente a EfficientNet-B0.
- EfficientNet-B0 ofrece una alternativa más barata en tiempo, pero con pérdida de rendimiento respecto al mejor modelo.
- ResNet50 no es comparable en igualdad de condiciones porque se entrenó solo 1 época; su resultado actual debe considerarse un baseline incompleto.

### Limitaciones y próximos pasos
- Aumentar profundidad de tuning para ResNet50 y EfficientNet-B0 con más épocas y estrategias de descongelado comparables.
- Repetir la mejor configuración de ConvNeXt-Base para estimar varianza y robustez.
- Añadir análisis por clase (matriz de confusión) para detectar errores sistemáticos.

Contenido recomendado
- Tabla comparativa final de modelos y configuraciones más relevantes.
- Interpretación de curvas para identificar underfitting, convergencia estable o sobreajuste.
- Análisis de errores por clase para detectar clases conflictivas o sesgos del dataset.
- Comentario sobre coste computacional y viabilidad práctica de la solución elegida.

Cierre esperado
- Indicar modelo final seleccionado y justificar la elección con datos.
- Explicar limitaciones del experimento (tiempo, número de corridas, desbalance, tamaño de muestra).
- Proponer mejoras futuras concretas (más tuning, augmentations adicionales, calibración o ensambles).

## Autoevaluación rápida
- Verde: completar >=80% con evidencia clara.
- Amarillo: mantener 60-79% e identificar faltantes en comparativas o reproducibilidad.
- Rojo: identificar <60% y planificar más runs y mejor justificación.
