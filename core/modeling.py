from __future__ import annotations

from dataclasses import dataclass
from threading import Event
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from core.cancellation import OperationCancelledError


@dataclass(frozen=True)
class ModelRunResult:
    response_name: str
    feature_names: list[str]
    train_size: int
    test_size: int
    r2: float
    rmse: float
    mae: float
    feature_importance: dict[str, float]


@dataclass(frozen=True)
class LinearModelRunResult:
    model_name: str
    response_name: str
    feature_names: list[str]
    train_size: int
    test_size: int
    r2: float
    rmse: float
    mae: float
    coefficients: dict[str, float]
    intercept: float


def train_random_forest(
    sample_table: pd.DataFrame,
    response_name: str,
    feature_names: Sequence[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[RandomForestRegressor, ModelRunResult]:
    return train_random_forest_incremental(
        sample_table=sample_table,
        response_name=response_name,
        feature_names=feature_names,
        test_size=test_size,
        random_state=random_state,
        progress_callback=None,
        cancel_event=None,
    )


def train_random_forest_incremental(
    sample_table: pd.DataFrame,
    response_name: str,
    feature_names: Sequence[str],
    test_size: float = 0.2,
    random_state: int = 42,
    progress_callback=None,
    cancel_event: Optional[Event] = None,
    total_estimators: int = 300,
    chunk_size: int = 25,
) -> tuple[RandomForestRegressor, ModelRunResult]:
    if response_name not in sample_table.columns:
        raise KeyError(f"Missing response column: {response_name}")

    missing_features = [name for name in feature_names if name not in sample_table.columns]
    if missing_features:
        raise KeyError(f"Missing feature columns: {missing_features}")

    X = sample_table.loc[:, list(feature_names)]
    y = sample_table.loc[:, response_name]

    _check_cancel(cancel_event)
    _emit_progress(progress_callback, 5, "Preparing train/test split")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    _check_cancel(cancel_event)
    model = RandomForestRegressor(
        n_estimators=0,
        random_state=random_state,
        n_jobs=-1,
        warm_start=True,
    )
    fitted_estimators = 0
    while fitted_estimators < total_estimators:
        _check_cancel(cancel_event)
        fitted_estimators = min(total_estimators, fitted_estimators + chunk_size)
        model.set_params(n_estimators=fitted_estimators)
        model.fit(X_train, y_train)
        progress_value = 10 + int(70 * (fitted_estimators / float(total_estimators)))
        _emit_progress(
            progress_callback,
            progress_value,
            "Training random forest ({0}/{1} trees)".format(fitted_estimators, total_estimators),
        )

    _check_cancel(cancel_event)
    _emit_progress(progress_callback, 88, "Evaluating random forest")
    predictions = model.predict(X_test)
    result = _build_model_run_result(
        model=model,
        response_name=response_name,
        feature_names=list(feature_names),
        train_size=len(X_train),
        test_size=len(X_test),
        y_test=y_test,
        predictions=predictions,
    )
    _emit_progress(progress_callback, 100, "Random forest complete")
    return model, result


def train_linear_regression(
    sample_table: pd.DataFrame,
    response_name: str,
    feature_names: Sequence[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[LinearRegression, LinearModelRunResult]:
    model = LinearRegression()
    return _train_linear_model(
        model=model,
        model_name="OLS",
        sample_table=sample_table,
        response_name=response_name,
        feature_names=feature_names,
        test_size=test_size,
        random_state=random_state,
    )


def train_ridge_regression(
    sample_table: pd.DataFrame,
    response_name: str,
    feature_names: Sequence[str],
    test_size: float = 0.2,
    random_state: int = 42,
    alpha: float = 1.0,
) -> tuple[Ridge, LinearModelRunResult]:
    model = Ridge(alpha=alpha, random_state=random_state)
    return _train_linear_model(
        model=model,
        model_name="Ridge",
        sample_table=sample_table,
        response_name=response_name,
        feature_names=feature_names,
        test_size=test_size,
        random_state=random_state,
    )


def _train_linear_model(
    model: LinearRegression,
    model_name: str,
    sample_table: pd.DataFrame,
    response_name: str,
    feature_names: Sequence[str],
    test_size: float,
    random_state: int,
) -> tuple[LinearRegression, LinearModelRunResult]:
    if response_name not in sample_table.columns:
        raise KeyError(f"Missing response column: {response_name}")

    missing_features = [name for name in feature_names if name not in sample_table.columns]
    if missing_features:
        raise KeyError(f"Missing feature columns: {missing_features}")

    X = sample_table.loc[:, list(feature_names)]
    y = sample_table.loc[:, response_name]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    result = LinearModelRunResult(
        model_name=model_name,
        response_name=response_name,
        feature_names=list(feature_names),
        train_size=len(X_train),
        test_size=len(X_test),
        r2=float(r2_score(y_test, predictions)),
        rmse=float(np.sqrt(mean_squared_error(y_test, predictions))),
        mae=float(mean_absolute_error(y_test, predictions)),
        coefficients={
            name: float(coef) for name, coef in zip(feature_names, model.coef_)
        },
        intercept=float(model.intercept_),
    )
    return model, result


def _build_model_run_result(
    model: RandomForestRegressor,
    response_name: str,
    feature_names: list[str],
    train_size: int,
    test_size: int,
    y_test: pd.Series,
    predictions: np.ndarray,
) -> ModelRunResult:
    return ModelRunResult(
        response_name=response_name,
        feature_names=feature_names,
        train_size=train_size,
        test_size=test_size,
        r2=float(r2_score(y_test, predictions)),
        rmse=float(np.sqrt(mean_squared_error(y_test, predictions))),
        mae=float(mean_absolute_error(y_test, predictions)),
        feature_importance={
            name: float(score) for name, score in zip(feature_names, model.feature_importances_)
        },
    )


def _emit_progress(progress_callback, percent: int, message: str) -> None:
    if progress_callback is not None:
        progress_callback(percent, message)


def _check_cancel(cancel_event: Optional[Event]) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise OperationCancelledError("Operation cancelled.")
