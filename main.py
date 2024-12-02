import mlflow
import os
import hydra
from omegaconf import DictConfig, OmegaConf


# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # You can get the path at the root of the MLflow project with this:
    root_path = hydra.utils.get_original_cwd()

    # Check which steps we need to execute
    if isinstance(config["main"]["execute_steps"], str):
        # This was passed on the command line as a comma-separated list of steps
        steps_to_execute = config["main"]["execute_steps"].split(",")
    else:
        assert isinstance(config["main"]["execute_steps"], list)
        steps_to_execute = config["main"]["execute_steps"]

    # Download step
    if "download" in steps_to_execute:

        _ = mlflow.run(
            os.path.join(root_path, "download"),
            "main",
            # Los parámetros del file_url están definidos en el archivo config.yaml
            parameters={
                "file_url": config["data"]["file_url"],
                "artifact_name": "raw_data.parquet",
                "artifact_type": "raw_data",
                "artifact_description": "Data as downloaded"
            },
        )

    if "preprocess" in steps_to_execute:

        _ = mlflow.run(
            os.path.join(root_path, "preprocess"),
            "main",
            parameters={
                # El input es la salida del download
                "input_artifact": "raw_data.parquet",
                "artifact_name": "processed_data.csv",
                "artifact_type": "processed_data",
                "artifact_description": "Data with preprocessing applied"
            },
        )

    if "check_data" in steps_to_execute:

        _ = mlflow.run(
            os.path.join(root_path, "check_data"),
            "main",
            parameters={
                # config.yaml
                "reference_artifact": config["data"]["reference_dataset"],
                # El input es la salida del download
                "sample_artifact": "processed_data.csv:latest",
                # config.yaml
                "ks_alpha": config["data"]["ks_alpha"],
            },
        )

    if "segregate" in steps_to_execute:

        _ = mlflow.run(
            os.path.join(root_path, "segregate"),
            "main",
            parameters={
                # El input es la salida del check_data
                "input_artifact": "preprocessed_data.csv:latest",
                # Name for the W&B artifact that will be created
                # This step will provide 2 artifacts: train_{artifact_root}.csv and test_{artifact_root}.csv
                "artifact_root": "data",
                "artifact_type": "segregated_data",
                # config.yaml
                "test_size": config["data"]["test_size"],
                "random_state": config["data"]["random_state"],
                "stratify": config["data"]["stratify"],
            },
        )

    if "random_forest" in steps_to_execute:

        # Para no tener que cargar todos los parámetros del random forest uno a uno, hacemos esto
        model_config = os.path.abspath("random_forest_config.yml")

        with open(model_config, "w+") as fp:
            fp.write(OmegaConf.to_yaml(config["random_forest_pipeline"]))

        _ = mlflow.run(
            os.path.join(root_path, "random_forest"),
            "main",
            parameters={
                # El input es la salida del segregate
                "train_data": "data_train.csv:latest",
                "model_config": model_config,
                "export_artifact": config["random_forest_pipeline"]["export_artifact"],
                "random_seed": config["main"]["random_seed"],
                "val_size": config["data"]["test_size"],
                "stratify": config["data"]["stratify"]
            },
        )

    if "evaluate" in steps_to_execute:

        _ = mlflow.run(
            os.path.join(root_path, "evaluate"),
            "main",
            parameters={
                # El input es la salida del random_forest
                "model_export": f"{config['random_forest_pipeline']['export_artifact']}:latest",
                "test_data": "data_test.csv:latest"
            },
        )


if __name__ == "__main__":
    go()
