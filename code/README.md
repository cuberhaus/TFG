# Guide

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
python3 src/optuna_train_model.py --debug
```

## Optimize Parameters with Raytune
```bash
python3 src/raytune_train_model.py --debug
```