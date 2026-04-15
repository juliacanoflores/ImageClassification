# TODO - Clasificación de Imágenes


# Orden de ejecución del proyecto

## 1) Reproducibilidad
- [ ] Fijar semillas (Python, NumPy, Torch, CUDA) — misma semilla para los 3 modelos.
- [ ] Registrar versiones de librerías y entorno en Lightning AI.
- [ ] Registrar hardware usado.

## 2) Model selection rationale
- [ ] Justificar los 3 modelos por familia arquitectónica, parámetros y coste compute.
- [ ] Definir métrica principal (val_accuracy) y métricas secundarias (F1 macro, tiempo/época).

## 3) Transfer-learning strategy (aplicar igual a los 3 modelos)
- [ ] Fase 1: entrenar solo la cabeza de clasificación (backbone congelado).
- [ ] Fase 2: descongelar las últimas capas/bloques para fine-tuning.
- [ ] Usar learning rate menor al descongelar (mínimo 10x menor).
- [ ] Documentar cuántas capas descongelar en cada modelo y por qué.
- [ ] Aplicar augmentations y normalización coherentes con los pesos preentrenados de cada modelo.
- [ ] Guardar el mejor checkpoint por validación en cada modelo.

## 4) Hyperparameter tuning (por cada uno de los 3 modelos)
- [ ] Definir el espacio de búsqueda antes de ejecutar runs.
- [ ] Probar varios learning rates (mínimo 3-4).
- [ ] Probar al menos 2 optimizadores o configuraciones equivalentes.
- [ ] Probar varios weight decay.
- [ ] Probar distintos niveles de descongelado (freeze layers).
- [ ] Probar distintos número de epochs.
- [ ] Ejecutar al menos 10 runs comparables por modelo (ideal: 12-20).

## 5) Tracking en W&B (en paralelo con 3 y 4)
- [ ] Usar naming convention clara: `{modelo}_{fase}_{lr}_{run_id}` y tags por modelo.
- [ ] Guardar config completa por run (modelo, lr, batch, optimizer, wd, epochs, freeze_layers, seed).
- [ ] Loguear train_loss, val_loss, val_accuracy por época.
- [ ] Loguear learning rate y tiempo por época.
- [ ] Subir artefactos del mejor checkpoint de cada modelo.
- [ ] Guardar curvas de entrenamiento, matriz de confusión y ejemplos de predicción por modelo.

## 6) Análisis final
- [ ] Tabla comparativa: mejor val_accuracy, F1 macro, tiempo/época y nº parámetros por modelo.
- [ ] Curvas de entrenamiento del mejor run de cada modelo (overfitting/underfitting).
- [ ] Matriz de confusión por modelo — analizar qué clases fallan y si el patrón es consistente.
- [ ] Selección del modelo final con criterio explícito (accuracy, coste, latencia de inferencia).
- [ ] Conclusión técnica: mejor modelo + por qué + coste operativo para producción.
- [ ] Guardar la configuración exacta del mejor run de cada modelo.
- [ ] Repetir el mejor run de cada modelo para verificar estabilidad.
- [ ] Mejoras futuras concretas.

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
