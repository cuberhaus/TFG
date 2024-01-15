OUTPUT_ZIP=TFG_Pol_Casacuberta.zip
SRC_DIR=./src

zip: $(SRC_DIR)
	powershell Compress-Archive -Force -Path $(SRC_DIR), README.md -DestinationPath $(OUTPUT_ZIP)

clean:
	powershell Remove-Item $(OUTPUT_ZIP)
