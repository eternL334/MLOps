# MLOps

## Цель проекта

Разработка системы поиска (dense retrieval) на основе претрейнов BERT моделей для задачи информационного поиска. Проект включает файнтюнинг модели на специализированном датасете и оценку качества работы системы.

## Набор данных

**FiQA (Financial Opinion Mining and Question Answering)**

- Домен: финансовые вопросы и ответы
- Размер корпуса: ~57,000 документов
- Обучающая выборка: ~5,500 запросов
- Dev выборка: ~500 запросов
- Test выборка: ~648 запросов
- Источник: BEIR benchmark

Датасет содержит финансовые вопросы пользователей и релевантные документы из финансовых форумов и новостных источников.

## Целевые метрики для продакшена

### Качество модели
- NDCG@10 ≥ 0.35
- Recall@10 ≥ 0.65
- MRR@10 ≥ 0.40

### Производительность (на сonsumer-grade видеокарте)
- Время развертывания модели (57k документов): ≤ 10 минут
- Время поиска на 1 запрос: ≤ 200 мс
- Throughput: ≥ 50 запросов/сек (batch size = 8)
- Использование GPU памяти: ≤ 12 GB
- Использование RAM: ≤ 24 GB

### Надежность
- Доля неуспешных запросов: ≤ 0.5%
- Uptime: ≥ 99%

## План экспериментов

1. Baseline (без дообучения)
2. Дообучение на train выборке
3. Подбор гиперпараметров
4. Эксперименты с другими моделями

## Установка

```bash
git clone https://github.com/eternL334/MLOps
cd MLOps
pip install -r requirements.txt

dvc repro # full pipeline

dvc repro prepare   # preprocessing
dvc repro train     # training
dvc repro evaluate  # evaluation

mlflow ui # for metrics ui

```

## Построение Docker образа


```bash
dvc repro # full pipeline

docker build -t ml-app:v1 . # build docker image

docker run --rm \
  -v "$(pwd)/test_input.csv":/data/input.csv \
  -v "$(pwd)/predictions":/data/output \
  -v "$(pwd)/data/fiqa":/data/corpus/fiqa \
  ml-app:v1 \
  --input_path /data/input.csv \
  --output_path /data/output/preds.csv \
  --raw_data_path /data/corpus/fiqa \
  --index_path /app/data/index/fiqa.index
```

## Запуск TorchServe

```bash
python -m src.dump_corpus


mkdir -p model_store

torch-model-archiver --model-name my-retriever \
  --version 1.0 \
  --serialized-file outputs/final_model/model.safetensors \
  --handler src/handler.py \
  --extra-files "outputs/final_model/config.json,outputs/final_model/retriever_config.json,outputs/final_model/tokenizer_config.json,outputs/final_model/tokenizer_config.json,outputs/final_model/vocab.txt,data/index/fiqa.index,data/index/fiqa.index.ids.pkl,data/corpus_map.pkl" \
  --export-path model_store


docker build -f Dockerfile.serve -t mymodel-serve:v1 .

docker run --rm -it -p 8080:8080 -p 8081:8081 mymodel-serve:v1

curl -X POST http://localhost:8080/predictions/my-retriever \
     -H "Content-Type: application/json" \
     -d '{"query": "What is inflation?"}'

```