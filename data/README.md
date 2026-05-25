# Data

This project is designed for MovieLens-25M, Amazon Product Reviews, or Yelp Open Dataset interaction data.

For local smoke tests, `data/samples/interactions.csv` contains a tiny implicit-feedback fixture with user, item, rating, timestamp, click, and session metadata.

Expected production schema:

```text
user_id,item_id,rating,timestamp,clicked,session_id,position
```

Recommended full-data path:

1. Download MovieLens-25M from https://grouplens.org/datasets/movielens/25m/
2. Place `ratings.csv` under `data/raw/movielens-25m/ratings.csv`.
3. Run `python3 scripts/prepare_movielens.py --input data/raw/movielens-25m/ratings.csv --output data/processed/interactions.csv`.

