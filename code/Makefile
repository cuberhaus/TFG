# Nombre del archivo zip de salida
OUTPUT_ZIP=TFG_Pol_Casacuberta.zip

# Directorio de la carpeta src del proyecto
SRC_DIR=./src

# Comando para comprimir los archivos en Windows con PowerShell
zip:
	powershell Compress-Archive -Path $(SRC_DIR) -DestinationPath $(OUTPUT_ZIP)

# Comando para limpiar (eliminar el archivo zip) en Windows con PowerShell
clean:
	powershell Remove-Item $(OUTPUT_ZIP)
