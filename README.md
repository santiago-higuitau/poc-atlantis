# Recomendación de Activos Financieros

## 1. Introducción

El sistema de recomendación de activos financieros tiene como objetivo sugerir productos financieros a los clientes basándose en su comportamiento financiero y características. En este documento, exploramos las fases clave del análisis exploratorio de datos (EDA) y los enfoques de clustering, similitud de coseno y el enfoque híbrido utilizado para optimizar las recomendaciones.

### Estructura del Documento

1. Análisis Exploratorio de Datos (EDA)
2. Enfoque de Clustering con K-Means
3. Similitud de Coseno
4. Enfoque Híbrido: Clustering y Similitud

## 2. Análisis Exploratorio de Datos (EDA)

### Descripción de los Datos

Los datos utilizados consisten en:
- **Activos Financieros:** Tarjetas de crédito, préstamos, vivienda, vehículos.
- **Comportamiento Transaccional:** Frecuencia y monto total de las transacciones por cliente.
- **Pasivos:** Saldos en cuentas de ahorro, cheques, y CDP (Certificado de Depósito a Plazo).

### Transformaciones y Limpieza

Se realizaron las siguientes transformaciones para preparar los datos para el análisis:
- **Normalización:** Los datos financieros, como montos de transacción y saldos, se normalizaron para eliminar sesgos por rangos dispares.
- **Imputación de valores nulos:** Se imputaron los valores faltantes con ceros en los casos donde no se registraban transacciones o pasivos.
- **Transformación binaria:** Los activos financieros se transformaron en valores binarios (1 si el cliente posee el activo, 0 si no lo posee).

### Visualizaciones

Se crearon visualizaciones para comprender la distribución de los datos. Entre ellas:
- **Distribución de Frecuencia de Transacciones:** Para observar la variabilidad en el comportamiento de transacciones entre clientes.
- **Distribución de Saldos de Ahorro:** Para identificar patrones en la tenencia de cuentas de ahorro.

## 3. Enfoque de Clustering con K-Means

Se utilizó el algoritmo K-Means para agrupar a los clientes:
1. **Normalización de Características:** Las características de los clientes fueron normalizadas para eliminar el sesgo.
2. **Número Óptimo de Clusters:** Se usó el método del codo y silueta para determinar el número óptimo de clústeres, seleccionando 3 como la cantidad ideal.
3. **Asignación de Clústeres:** Los clientes fueron asignados a uno de los 3 clústeres:
   - **Cluster 0:** Clientes con una combinación balanceada de activos financieros (tarjetas de crédito, préstamos de consumo, productos de vivienda) y comportamiento transaccional moderado.
   - **Cluster 1:** Clientes con un uso intensivo de productos financieros específicos (como tarjetas de crédito y préstamos) y alta actividad transaccional.
   - **Cluster 2:** Clientes con menos productos financieros y con comportamientos financieros más conservadores (menos transacciones, saldos más bajos).

### Visualización de Clusters

Se utilizó PCA (Análisis de Componentes Principales) para reducir las dimensiones de los datos y visualizar los clústeres en un espacio bidimensional. Esta técnica permitió entender cómo los clientes se agrupan en diferentes segmentos.

## 4. Similitud de Coseno

### Implementación

Dentro de cada clúster, se calculó la similitud de coseno para encontrar los clientes más similares entre sí. Esto permitió que el sistema de recomendación sugiera productos financieros que clientes similares ya poseen.

### Proceso de Recomendación

1. **Similitud dentro de Clústeres:** Para cada cliente, se calculó la similitud de coseno con otros clientes del mismo clúster.
2. **Filtrado de Activos:** Los activos que no poseía el cliente pero que eran comunes entre clientes similares fueron recomendados.

## 5. Enfoque Híbrido: Clustering y Similitud de Coseno

Combinar clustering con similitud de coseno mejora la precisión de las recomendaciones. El clustering agrupa a los clientes basándose en patrones generales, mientras que la similitud de coseno afina la recomendación considerando las similitudes individuales dentro de esos grupos.

### Implementación del Enfoque Híbrido

1. **Clustering Inicial:** Primero, se agruparon los clientes en clústeres utilizando K-Means.
2. **Similitud de Coseno dentro del Clúster:** Para cada cliente en un clúster, se calculó la similitud de coseno con otros clientes del mismo clúster.
3. **Recomendación de Activos:** Basado en la similitud, se recomendaron activos financieros que otros clientes similares poseían.

### Ventajas del Enfoque Híbrido

- **Personalización:** Las recomendaciones son más precisas al basarse en características compartidas con clientes similares.
- **Escalabilidad:** Este enfoque es eficiente y escalable, permitiendo recomendaciones precisas incluso para grandes volúmenes de clientes.
