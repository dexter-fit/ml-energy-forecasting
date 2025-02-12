# %%
import keras
import keras.api.layers as layers
from pathlib import Path
from preprocess_dataset import frhouse_multivariate
from evaluation import eval_forecast, average_metrics
import numpy as np


# %%
def eval_transfer(models: None|list[tuple[str|Path, list[str]]] = None, architecture: str|None = None):
    MODELS_ROOT = Path(__file__).parent.parent.parent / "checkpoints"
    RESULTS = MODELS_ROOT.parent / "results"
    RESULTS.mkdir(parents=True, exist_ok=True)

    models = models if models is not None else [
            (MODELS_ROOT / "tcn-lstm-nist-[history_temperature]-2025-02-12 14:26:47.864516" / "tcn-lstm-nist-[history_temperature]-2025-02-12 14:26:47.864516.keras", ["temperature"])
            # (MODELS_ROOT / "tcn-twohead-res-500.keras", ["temperature"]),
            # (MODELS_ROOT / "tcn-wind-res-500.keras", ["wind"]),
            # (MODELS_ROOT / "tcn-humidity-res-500.keras", ["humidity"]),
            # (MODELS_ROOT / "tcn-all-weather-res-500.keras", ["temperature", "humidity",  "wind"]),
    ]

    architecture = architecture if architecture is not None else "tcn" # "tcn-lstm"

    for model_path, vars in models:
        model = keras.models.load_model(model_path)
        X_train, X_exo_list_train, y_train, X_test, X_exo_list_test, y_test = frhouse_multivariate(None, vars)

        if architecture == "tcn-lstm":
            for i in range(len(X_exo_list_train)):
                X_train = np.concatenate((X_train, X_exo_list_train[i]), axis=-1) # type: ignore
            for i in range(len(X_exo_list_test)):
                X_test = np.concatenate((X_test, X_exo_list_test[i]), axis=-1) # type: ignore

        if architecture == "tcn-lstm":
            y_hat = model.predict(X_train) # NOTE: can use X_train, since we are not training, so we never saw it # type: ignore
            y_hat = y_hat.squeeze(axis=-1)
        else:
            y_hat = model.predict([X_train, *X_exo_list_train]) # type: ignore

        eval_res = eval_forecast(y=y_train, y_hat=y_hat)

        rmse, mape, nrmse = average_metrics(eval_res)


        eval_lines = [f"rmse: {rmse}\n",
                      f"mape: {mape}\n",
                      f"nrmse: {nrmse}\n"]

        print(f"model: {model_path}")
        print("".join(eval_lines))
        model_name = Path(model_path).name
        with open(RESULTS / f"ihepc-transfer-results-{model_name}.res", "w") as f:
            f.write(f"finetuning on IHEPC: 6mo training, 12mo test\n")
            f.write(f"variables: {vars}")
            f.write(f"model: {model_name}\n")
            f.writelines(eval_lines)


if __name__ == "__main__":
    eval_transfer()
    exit()
