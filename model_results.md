# Informe de Resultados del Modelo YOLOv8

## Configuración del Entrenamiento

- **Modelo Base**: YOLOv8s (small)
- **Tarea**: Detección de objetos
- **Tamaño de imagen**: 640x640
- **Épocas totales**: 398
- **Tiempo total**: 17,014.1 segundos (~4.7 horas)
- **Batch size**: Automático (-1)
- **Optimizador**: Auto
- **Device**: GPU (Kaggle)

### Hiperparámetros Principales

- Learning rate inicial (lr0): 0.01
- Learning rate final (lrf): 0.000114438
- Momentum: 0.937
- Weight decay: 0.0005
- Warmup epochs: 3.0

### Aumentación de Datos

- Rotación horizontal (flip): 50% de probabilidad
- Escala: ±50%
- Traslación: ±10%
- Mosaic: Activado
- Random Erasing: 40%
- Auto Augment: RandAugment

## Resultados del Entrenamiento

### Métricas Finales (Época 398)

- **Precisión**: 98.486%
- **Recall**: 95.420%
- **mAP50**: 97.065%
- **mAP50-95**: 94.582%

### Evolución del Entrenamiento

1. **Fase inicial** (Épocas 1-100):
   - Precisión mejoró de 42.591% a 97.281%
   - Recall aumentó de 47.393% a 93.057%
   - mAP50 creció de 40.422% a 96.918%
   - mAP50-95 aumentó de 21.513% a 84.769%
   - Learning rate inicial: 0.000546557

2. **Fase intermedia** (Épocas 101-250):
   - Precisión consistentemente sobre 97%
   - Recall estable por encima del 94%
   - mAP50 mantenido sobre 96%
   - mAP50-95 superó el 90%
   - Learning rate: 0.00128437 → 0.000503206
   - Estabilización notable en todas las métricas

3. **Fase final** (Épocas 251-398):
   - Precisión máxima de 98.486%
   - Recall alcanzó 95.420%
   - mAP50 alcanzó 97.065%
   - mAP50-95 llegó a 94.582%
   - Learning rate final: 0.000114438
   - Refinamiento fino de todas las métricas

### Pérdidas (Evolución época 1 → 398)

- Box loss: 1.28516 → 0.26532 (79% de reducción)
- Clasificación loss: 2.32056 → 0.20451 (91% de reducción)
- DFL loss: 1.42982 → 0.86165 (40% de reducción)

## Visualización de Resultados

### Curvas de Rendimiento

#### Curvas de Entrenamiento
![Resultados de Entrenamiento](model/train_results/results.png)
*Evolución de métricas durante el entrenamiento*

#### Curvas de Precisión y Recall
![Curva Precisión-Recall](model/test_results/BoxPR_curve.png)
*Curva Precisión-Recall que muestra el balance entre ambas métricas*

![Curva F1](model/test_results/BoxF1_curve.png)
*Curva F1 que indica el rendimiento general del modelo*

### Matriz de Confusión
![Matriz de Confusión](model/test_results/confusion_matrix.png)
*Matriz de confusión que muestra el rendimiento del modelo por clase*

### Ejemplos de Detección
![Ejemplo de Detección](model/test_results/val_batch0_pred.jpg)
*Ejemplo de detecciones realizadas por el modelo en el conjunto de validación*




---
*Las gráficas y métricas mostradas provienen de los archivos de resultados generados durante el entrenamiento y evaluación del modelo.*