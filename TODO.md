# TODO - Clasificación de Imágenes


# Por hacer
# Checklist de calidad experimental (Transfer Learning + W&B)

## 1) Model selection rationale
- [ ] Definir objetivo y métrica principal (ej. val_accuracy).
- [ ] Comparar al menos 2-3 modelos preentrenados (ej. ResNet50, EfficientNet-B0, ConvNeXt-Base).
- [ ] Justificar cada candidato por precisión esperada, coste y tiempo de entrenamiento.
- [ ] Registrar resultados comparables: mejor val_accuracy, tiempo por época, tamaño del modelo.
- [ ] Elegir el modelo final con criterio explícito (no por intuición).

## 2) Transfer-learning strategy
- [ ] Fase 1: entrenar solo la cabeza de clasificación (backbone congelado).
- [ ] Fase 2: descongelar las últimas capas/bloques para fine-tuning.
- [ ] Usar learning rate menor al descongelar.
- [ ] Documentar cuántas capas descongelar y por qué.
- [ ] Aplicar augmentations y normalización coherentes con el modelo preentrenado.
- [ ] Guardar el mejor checkpoint según validación.

## 3) Hyperparameter tuning depth
- [ ] Definir un espacio de búsqueda antes de ejecutar runs.
- [ ] Probar varios learning rates (mínimo 3-4).
- [ ] Probar al menos 2 optimizadores o configuraciones equivalentes.
- [ ] Probar varios weight decay.
- [ ] Probar distintos batch sizes (si hardware lo permite).
- [ ] Probar distintos niveles de descongelado.
- [ ] Ejecutar al menos 10 runs comparables (ideal: 12-20).
- [ ] Reportar tendencias (qué funcionó y qué no), no solo el mejor run.

## 4) Calidad de tracking en W&B
- [ ] Definir nombre claro y tags útiles para cada run.
- [ ] Guardar config completa por run (modelo, lr, batch, optimizer, wd, epochs, seed).
- [ ] Loguear train_loss, val_loss, val_accuracy por época.
- [ ] Loguear learning rate y tiempo por época.
- [ ] Subir artefactos del mejor modelo.
- [ ] Guardar visualizaciones clave (curvas, confusión, ejemplos de predicción).

## 5) Reproducibilidad
- [ ] Fijar semillas (Python, NumPy, Torch, CUDA).
- [ ] Registrar versiones de librerías y entorno.
- [ ] Registrar hardware usado.
- [ ] Guardar la configuración exacta del mejor run.
- [ ] Repetir el mejor experimento para verificar estabilidad.

## 6) Análisis final
- [ ] Incluir tabla comparativa final de modelos y configuraciones.
- [ ] Explicar overfitting/underfitting con curvas de entrenamiento.
- [ ] Analizar errores por clase (matriz de confusión o equivalente).
- [ ] Cerrar con conclusión técnica clara: mejor modelo + por qué + coste.
- [ ] Añadir mejoras futuras concretas.

## Autoevaluación rápida
- [ ] Verde: completar >=80% con evidencia clara.
- [ ] Amarillo: mantener 60-79% e identificar faltantes en comparativas o reproducibilidad.
- [ ] Rojo: identificar <60% y planificar más runs y mejor justificación.


# En progreso
- [ ] (Añadir tarea aquí) | Responsable: 

# Completado
- [x] Linkar WeightAndBiases al proyecto | Responsable: Julia
- [x] Conectar el modelo a la API | Responsable: Julia
- [x] Conectar la API al frontend | Responsable: Julia
- [x] API documentation | Responsable: Javi
- [x] Añadir endpoints a la API| Responsable: Javi
- [x] Actualizar frontend para añadir nuevos endpoints | Responsable: Javi
- [x] Añadir documentación swagger y estática de la API | Responsable: Javi
