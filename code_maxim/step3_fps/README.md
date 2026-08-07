# Step 3 — sampled softmax с in-batch negatives

## Что изучаем

Для каждого запроса модель должна выбрать кликнутый баннер среди небольшого
набора кандидатов. Набор строится прямо из batch:

- batch содержит 512 положительных пар `(query, clicked_banner)`;
- баннер на той же позиции — positive;
- остальные 511 баннеров — sampled negatives;
- `log q(banner)` correction пока не используется.

Матрица logits имеет размер `512 × 512`:

```python
logits = query_vectors @ banner_vectors.T / temperature
labels = arange(512)
loss = cross_entropy(logits, labels)
```

В отличие от step2, loss считается только в направлении `query → banner`.
Step2 дополнительно усреднял его с обратным `banner → query` CE.

## Подготовка данных

YQL читает полный train, оставляет только клики и заранее превращает текстовые
фичи в `uint32`-хеши:

```text
input:  //home/bm/users/argus/mla/data/train_100m
output: //home/bm/users/argus/mla/steps/step3_fps/train_100m
```

Запрос: `code/step3_fps/prepare_dataset.yql`.

В таблице остаются только два столбца:

- `query_features`: хеши query tokens и RegionID;
- `banner_features`: хеши BannerID, BannerTitle и BannerText.

После пересчёта `train_100m` фактическое число clicked pairs нужно проверить по
`@row_count` подготовленной таблицы. Обучение идёт одну эпоху с batch size 512.

## Обучение

```bash
python code/step3_fps/train_yt.py \
  --underdeep-project mla \
  --underdeep-experiment maxim-kuzin
```

Batch size зафиксирован равным 512. Данные читаются напрямую из YT.

После каждых примерно 20 000 train-строк модель считается на одной и той же
псевдослучайной выборке 2 000 кликов из полного `val`. В UnderDeep появляются
кривые:

- `val/loss`;
- `val/in_batch_accuracy`.

YQL выборки: `code/step3_fps/prepare_validation.yql`. Он оставляет только
клики и пишет их в:

```text
//home/bm/users/argus/mla/steps/step3_fps/val_clicks_2k
```

Фиксированная выборка делает validation-кривые разных запусков сравнимыми.

Артефакт:

```text
~/mla/data/artifacts/step3_fps/
```

Candidate embeddings всегда строятся из канонического
`~/mla/data/index_1m.parquet`. Переэкспорт без повторного обучения:

```bash
python code/step3_fps/export_candidates.py
```

## Viewer и метрики

```bash
python common/viewer.py \
  --code code/step3_fps/inference.py \
  --artifact-dir ~/mla/data/artifacts/step3_fps \
  --host :: \
  --port 8083
```

```bash
python common/evaluate.py \
  --code code/step3_fps/inference.py \
  --artifact-dir ~/mla/data/artifacts/step3_fps \
  --val-file ~/mla/data/val_clicks.parquet
```

Следующее естественное улучшение — учесть частоту попадания баннера в набор
негативов и вычесть `log q(banner)` из его logit.
