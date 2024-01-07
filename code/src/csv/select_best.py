import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def seleccionar_mejor_configuracion(archivo_csv, archivo_salida):
    # Cargar los datos desde un archivo CSV
    datos = pd.read_csv(archivo_csv)

    # Suponiendo que las métricas clave son AP_50_95_all y AR_50_95_all_maxDets_100
    # Ajusta estos nombres de columna según tus datos
    metrica_ap = 'AP_50_95_all'
    metrica_ar = 'AR_50_95_all_maxDets_100'

  	# Normalizar las métricas clave
    scaler = MinMaxScaler()

    datos[[metrica_ap, metrica_ar]] = datos[[metrica_ap, metrica_ar]]

    print(datos[[metrica_ap, metrica_ar]])
    print(datos[metrica_ap] + datos[metrica_ar])

    # datos[[metrica_ap, metrica_ar]] = scaler.fit_transform(datos[[metrica_ap, metrica_ar]])

    # Calcular el F1-score
    datos['F1_score'] = 2 * (datos[metrica_ap] * datos[metrica_ar]) / (datos[metrica_ap] + datos[metrica_ar])
    print(datos['F1_score'])

    # Seleccionar la configuración con el F1-score más alto
    mejor_configuracion = datos.loc[datos['F1_score'] == max(datos['F1_score'])]

    # Seleccionar las primeras 6 columnas y la columna F1_score
    datos_reducidos = datos.iloc[:, :6]
    datos_reducidos['F1_score'] = datos['F1_score']

    # Guardar estas columnas en un archivo CSV
    datos_reducidos.to_csv(archivo_salida, index=False)

    # # Seleccionar la configuración con el mejor rendimiento en AP y AR
    # mejor_configuracion = datos.loc[datos[metrica_ap] + datos[metrica_ar] == max(datos[metrica_ap] + datos[metrica_ar])]

    return mejor_configuracion

# Llamar a la función con el nombre de tu archivo CSV
nombre_archivo_csv = 'model_performances.csv'  # Cambia esto por el nombre de tu archivo
archivo_salida='f_score.csv'
mejor_configuracion = seleccionar_mejor_configuracion(nombre_archivo_csv, archivo_salida=archivo_salida)
print(mejor_configuracion.iloc[:, 0:6])  # Imprimir las primeras 6 columnas de la configuración
