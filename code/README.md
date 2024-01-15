# Estructura
En la carpeta src encontramos:
- **clases:** Contiene el código principal del proyecto a partir de este se hacen llamadas a estos paquetes para hacer experimentos generalmente desde los scripts localizados en src/
- **csv**: Contiene los scripts para formatear los resultados para añadirlos a la memoria.
- **j_notebooks**: Contiene notebooks interactivos que sirven para ver visualmente los resultados de forma rapida y no tener que re-ejecutar código, especialmente útil para el análisis de datos.
- **python_scripts**: Contiene scripts de python que generalmente solo se ejecutan una vez o se ejecutan raramente.
- **shell_scripts**: Encontramos scripts para automatizar procesos o para facilitar tareas.

# Resumen
Este proyecto tiene como objetivo investigar y desarrollar modelos de aprendizaje profundo generativos para la traducción de imagen a imagen. En concreto el objetivo es mejorar el entrenamiento de modelos de aprendizaje profundo para la detección de pólipos en colonoscopias. El desafío en el entrenamiento de este tipo de problemas médicos suele ser que los conjuntos de datos existentes tienen limitaciones como la escasez, diversidad o calidad de los datos. 
El objetivo es aprovechar las capacidades de los modelos generativos, para generar datos sintéticos para complementar los conjuntos de datos existentes para abordar sus limitaciones.
En este caso se ha utilizado el conjunto de datos LDPolypVideo que consiste en videos de colonoscopias y contiene máscaras que serán utilizadas para entrenar modelos.
La detección de objetos en concreto de pólipos ya se ha hecho con anterioridad, pero en este trabajo se ha buscado mejorar la precisión con la que se detectan los pólipos, evitando falsos positivos y falsos negativos. También se han investigado diferentes posibles soluciones al problema de la generación de imagen.

# Abstract
This project aim is to investigate and develop machine learning models for image-to-image translation. The main goal is to improve the training of machine learning models in the task of polyp detection in colonoscopies. The main challenge in this kind of medical problem tends to be limitations on the scarcity, diversity and quality of the data. The objective is to take advantage of the capabilities of generative models to generate synthetic data to improve the qualities of the existing dataset. 
In this case the dataset LDPolypVideo was used. It's made up of colonoscopy's videos, and it also contains masks (bounding boxes) which will be used to train the models. The detection of objects in particular of polyps is nothing new, but in this work we aim to improve the precision and avoid false positives and false negatives. We have also researched possible solutions to the problem of generative AI in images.

### Palabras clave
Aprendizaje automático, Visión por computador, Modelos generativos, traducción de imagen a imagen

### Keywords
Machine learning, Computer vision, Modelos generativos, traducción de imagen a imagen

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