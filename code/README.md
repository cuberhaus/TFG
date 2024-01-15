# Estructura
En la carpeta src encontramos:
- **clases:** Contiene el código principal del proyecto a partir de este se hacen llamadas a estas funciones para hacer experimentos generalmente desde los scripts localizados en src/
- **csv**: Contiene los scripts para formatear los resultados para añadirlos a la memoria.
- **j_notebooks**: Contiene notebooks interactivos que sirven para ver visualmente los resultados de forma rapida y no tener que re-ejecutar código, especialmente útil para el análisis de datos.
- **python_scripts**: Contiene scripts de python que generalmente solo se ejecutan una vez o se ejecutan raramente.
- **shell_scripts**: Encontramos scripts para automatizar procesos o para facilitar tareas.

# Guide

A continuación encontramos una guía con los comandos que se han usado con más frecuencia durante la elaboración del proyecto.

## Train models
```bash
python3 src/train_models.py configurations/model.json --debug
```

## Join code
```bash
python src/python_scripts/join_code.py
```

## Optimize Parameters with Optuna
```bash
python3 optuna_train_model.py --debug
nohup python3 optuna_train_model.py FasterRCNN > optuna_train_model.log &
nohup python3 optuna_train_model.py RetinaNet > optuna_train_model.log &
nohup python3 optuna_train_model.py SSD > optuna_train_model.log &
```

## Optimize Parameters with Raytune
```bash
python3 src/raytune_train_model.py --debug
```

## Train the CycleGAN model
```bash
python3 src/cyclegan_train.py
nohup python3 cyclegan_train.py > cyclegan_train.log &
```

## Test the CycleGAN model
```bash
python3 src/cyclegan_test.py
nohup python3 cyclegan_test.py > cyclegan_test.log &
```

## Create masks from bounding boxes
```bash
python3 src/create_masks.py
```

## Copy files to cycle GAN structure
```bash
python3 src/copy_files_to_cyclegan_structure.py
```

## Evaluate all models
```bash
python3 src/evaluate_models.py 
nohup python3 evaluate_models.py > evaluate_models.log &
```

## Sync models
From remote to local
```
rsync -avz --progress casacuberta@teegarden.cs.upc.edu:/home/casacuberta/TFG/old/saved_models /home/pol/TFG/code/tmp
```

## Scp file
```
 scp casacuberta@teegarden.cs.upc.edu:/home/casacuberta/TFG/code/out/model_performances.csv C:\Users\pol\repos\TFG\code\tmp
```

## To generate environment.yml from conda environment with the dependencies:
```
conda env export > environment.yml
```

## To create a Conda environment with those packages:
```
conda env create -f environment.yml
```