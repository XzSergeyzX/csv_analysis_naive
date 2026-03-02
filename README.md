# Core 1

Basic data prep + baseline ML pipeline (train/predict).

## Установка
```bash
pip install -r requirements.txt
```

## Обучение
```bash
python src/train.py --config configs/config.yaml
```

## Инференс
```bash
python src/predict.py \
  --config configs/config.yaml \
  --input data/raw/input.csv \
  --output data/out/preds.csv
```