import streamlit as st
import pandas as pd

# Título de la aplicación
st.title("Recomendaciones de Activos Financieros - Enfoque Híbrido KMeans + Filtrado")

# Descripción del campo "Recommendation Value"
st.write("""
### Descripción del Recommendation Value:
El **Recommendation Value** indica la confianza o relevancia de cada activo recomendado para un cliente.
Este valor puede estar basado en la similitud con otros clientes en el mismo clúster o en la frecuencia de aparición de ese activo en clientes similares.
Un valor más alto significa que es más probable que el activo recomendado sea relevante para el cliente.
""")

# Descripción de los clústeres generados por KMeans (con k=3)
st.write("""
### Descripción de los Clústeres Generados por KMeans:
Los **clusters generados por KMeans** agrupan a los clientes según sus características financieras y demográficas.
Cada cliente se asigna a uno de estos 3 clústeres basados en similitudes de:
- **Activos Financieros**: Tarjetas de crédito, préstamos, vivienda, vehículos.
- **Comportamiento Transaccional**: Frecuencia y monto de las transacciones.
- **Saldos en Cuentas**: Ahorros, cheques, CDP.

#### Interpretación de los Clústeres:
- **Cluster 0**: Clientes con una combinación balanceada de activos financieros (tarjetas de crédito, préstamos de consumo, productos de vivienda) y comportamiento transaccional moderado.
- **Cluster 1**: Clientes con un uso intensivo de productos financieros específicos (como tarjetas de crédito y préstamos) y alta actividad transaccional.
- **Cluster 2**: Clientes con menos productos financieros y con comportamientos financieros más conservadores (menos transacciones, saldos más bajos).
""")

# Cargar los datos desde la ubicación específica
file_path = './data/data_processed/results_hybrid_approach_kmeans_filter_v3.parquet'

# Leer el archivo Parquet directamente
try:
    df = pd.read_parquet(file_path)
    st.success("Archivo cargado correctamente.")

    # Mostrar un resumen de los datos
    st.subheader("Resumen de los Datos")
    st.dataframe(df)
    
    # Mostrar información general
    st.write("Total de clientes con recomendaciones efectivas:", len(df))
    
    # Filtrar por cliente específico si es necesario
    cliente_id = st.text_input("Buscar por ID de Cliente (NEW_BP)")
    
    if cliente_id:
        filtered_df = df[df['NEW_BP'].astype(str).str.contains(cliente_id)]
        st.write(f"Clientes encontrados: {len(filtered_df)}")
        st.dataframe(filtered_df)
    
    # Mostrar las recomendaciones por producto
    st.subheader("Distribución de Recomendaciones por Activo")
    recommended_assets_count = df['Recommended_Assets'].explode().value_counts()
    st.bar_chart(recommended_assets_count)
    
    # Mostrar la distribución de los clientes por clúster
    st.subheader("Distribución de Clientes por Clúster")
    cluster_distribution = df['Cluster'].value_counts()
    st.bar_chart(cluster_distribution)
    
    # Mostrar la imagen guardada de los clusters en 2D
    st.subheader("Clusters de Clientes en PCA 2D")
    st.image('./data/data_processed/clusters_pca_2d.png', caption='Clusters de Clientes en Espacio PCA 2D', use_column_width=True)
    
    # Descargar los resultados filtrados si se busca un cliente
    if cliente_id and not filtered_df.empty:
        st.download_button("Descargar Recomendaciones Filtradas", data=filtered_df.to_csv(), file_name=f'recomendaciones_{cliente_id}.csv')

except FileNotFoundError:
    st.error(f"No se pudo encontrar el archivo en la ruta: {file_path}")

except Exception as e:
    st.error(f"Error al cargar el archivo: {e}")
