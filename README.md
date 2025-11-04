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
# Установка зависимостей
pip install -r requirements.txt

# Finetune and evaluation
python main.py --config config.yaml 

# Only evaluation of pretrained model (from config)
python main.py --config config.yaml --eval-only

# Only evaluation of finetuned model (from path)
python main.py --config config.yaml --eval-only --model-path ./outputs/final_model
```