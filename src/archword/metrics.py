"""Метрики качества: ранговая корреляция, ошибки, согласие разметчиков."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score


def regression_report(y_true, y_pred) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    constant = np.ptp(y_true) == 0 or np.ptp(y_pred) == 0  # корреляция не определена
    rho = 0.0 if constant else spearmanr(y_true, y_pred)[0]
    return {
        "spearman": round(float(rho), 4),
        "rmse": round(float(np.sqrt(np.mean((y_true - y_pred) ** 2))), 4),
        "mae": round(float(np.mean(np.abs(y_true - y_pred))), 4),
        "n": int(len(y_true)),
    }


def mean_baseline_report(y_train, y_test) -> dict[str, float]:
    """Модель «всегда предсказывать среднее»: нижняя планка для RMSE и MAE."""
    pred = np.full(len(y_test), float(np.mean(y_train)))
    return regression_report(y_test, pred)  # у константы корреляция равна 0 по определению


def to_bins(values, edges=(0.15, 0.3, 0.45, 0.6)) -> np.ndarray:
    """Перевод непрерывной оценки 0..1 в классы 1..5 для сравнения со шкалой разметки."""
    return np.digitize(np.asarray(values, float), edges) + 1


def agreement(a, b) -> dict[str, float]:
    """Согласие двух разметок по шкале 1..5: взвешенная каппа и корреляция."""
    a, b = np.asarray(a), np.asarray(b)
    return {
        "qwk": round(float(cohen_kappa_score(a, b, weights="quadratic")), 4),
        "spearman": round(float(spearmanr(a, b)[0]), 4),
        "n": int(len(a)),
    }
