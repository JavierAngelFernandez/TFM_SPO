import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, recall_score, precision_score
from catboost import CatBoostRegressor
from pnl_LR import LogSigmoidPnLLinear, ExpoPnLLinear, TanhPnLLinear
from pnl_custom_losses import PnLMetric, TanhPnLLoss, LogSigmoidPnLLoss, ExpoPnLLoss


primary_cols = [
    'date',
    'precio_spot_diario_Espana',
    'precio_spot_intradiario1_Espana',
    'precio_spot_intradiario2_Espana',
    'dem_prev_diaria_6',
    'gen_prev_eolica_6',
    'gen_prev_fotov_6',
    'spread_ida1_spot',
    'spread_ida2_spot',
]

lr_metric_map = {
    "TanhPnLLoss": TanhPnLLinear, 
    "LogSigmoidPnLLoss": LogSigmoidPnLLinear,
    "ExpoPnLLoss": ExpoPnLLinear,
}

metric_map = {
    "TanhPnLLoss": TanhPnLLoss, 
    "LogSigmoidPnLLoss": LogSigmoidPnLLoss,
    "ExpoPnLLoss": ExpoPnLLoss,
}


def crear_lags_horarios(df, columnas, lags):
    """Creamos lags horarios de las variables `columnas`"""

    for col in columnas:
        for h in lags:
            fechas_lag = df.index - pd.Timedelta(hours=h)
            df[f"{col}_lag_{h}h"] = df[col].reindex(fechas_lag).values

    return df


def get_ridge_model():
    """Obtener el regresor de una regresion Ridge"""
    ridge_pipeline = Pipeline(
        steps=[
            ("std_scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0))
        ]
    )
    model = TransformedTargetRegressor(
        regressor=ridge_pipeline, transformer=StandardScaler()
    )
    return deepcopy(model)


def get_lr_model(loss_function):
    """Obtener el regresor de una regresion Ridge"""
    lr = lr_metric_map[loss_function](learning_rate=1e-2, epochs=300)
    lr_pipeline = Pipeline(
        steps=[
            ("std_scaler", StandardScaler()),
            ("lr", lr)
        ]
    )
    model = TransformedTargetRegressor(
        regressor=lr_pipeline, transformer=StandardScaler()
    )
    return deepcopy(model)


def get_rf_model():
    """Obtener el regresor de una regresion Random Forest"""
    rf_pipeline = Pipeline(
        steps=[
            ("std_scaler", StandardScaler()),
            ("ridge", RandomForestRegressor(random_state=42, n_estimators=100, max_depth=10, min_impurity_decrease=0.001, min_samples_leaf=1))
        ]
    )
    model = TransformedTargetRegressor(
        regressor=rf_pipeline, transformer=StandardScaler()
    )
    return deepcopy(model)


def get_clsrf_model():
    """Obtener el regresor de una regresion Random Forest"""
    rf_pipeline = Pipeline(
        steps=[
            ("std_scaler", StandardScaler()),
            ("ridge", RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10, min_impurity_decrease=0.001, min_samples_leaf=1))
        ]
    )
    return deepcopy(rf_pipeline)


def get_catboost_model(loss_function=None):
    """Obtener el regresor de una regresion por Gradient Boosting"""
    if loss_function is None:
        model = CatBoostRegressor(verbose=0)
    else:
        model = CatBoostRegressor(
            loss_function=metric_map[loss_function](),
            eval_metric=PnLMetric(),
            verbose=0
        )
    return deepcopy(model)


def get_metrics_regression(df_results: pd.DataFrame, tgt_col: str, pred_cols: list):
    """Calcula el MAE, MSE y R2."""
    resultados = []
    for modelo in pred_cols:
        mae = mean_absolute_error(y_true=df_results[tgt_col], y_pred=df_results[modelo])
        mse = mean_squared_error(y_true=df_results[tgt_col], y_pred=df_results[modelo])
        r2 = r2_score(y_true=df_results[tgt_col], y_pred=df_results[modelo])

        resultados.append({
            "Modelo": modelo,
            "MAE": mae,
            "MSE": mse,
            "R2": r2
        })
    return pd.DataFrame(resultados)


def get_metrics_clasification(df_results: pd.DataFrame, tgt_col: str, pred_cols: list):
    """Calcula el MAE, MSE y R2."""
    resultados = []
    for modelo in pred_cols:
        y_pred = np.where(df_results[modelo] >= 0, 1, -1)
        y_true = np.where(df_results[tgt_col] >= 0, 1, -1)
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        recall = recall_score(y_true=y_true, y_pred=y_pred)
        prec = precision_score(y_true=y_true, y_pred=y_pred)

        resultados.append({
            "Modelo": modelo,
            "Accuracy": acc,
            "Recall": recall,
            "Precision": prec
        })
    return pd.DataFrame(resultados)


def procesa_datos(path="./data/datos_TFM.csv", cols_to_keep=primary_cols):
    """Procesamos los datos"""
    df =  pd.read_csv(path)

    ## calculamos spreads
    df.loc[:, "spread_ida1_spot"] = df["precio_spot_intradiario1_Espana"] - df["precio_spot_diario_Espana"]
    df.loc[:, "spread_ida2_spot"] = df["precio_spot_intradiario2_Espana"] - df["precio_spot_diario_Espana"]
    df.loc[:, "spread_ida3_spot"] = df["precio_spot_intradiario3_Espana"] - df["precio_spot_diario_Espana"]

    ## pasamos a formato fechas
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("Europe/Madrid")

    return df[cols_to_keep].set_index("date").dropna()



