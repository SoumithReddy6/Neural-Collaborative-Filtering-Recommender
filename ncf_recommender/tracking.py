from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator


@contextmanager
def mlflow_run(run_name: str, params: dict[str, object] | None = None) -> Iterator[None]:
    try:
        import mlflow

        with mlflow.start_run(run_name=run_name):
            if params:
                mlflow.log_params(params)
            yield
    except Exception:
        yield


def log_metrics(metrics: dict[str, float]) -> None:
    try:
        import mlflow

        mlflow.log_metrics(metrics)
    except Exception:
        return

