import numpy as np
from mlflow.types import TensorSpec
import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec
import tensorflow as tf
from urllib.parse import urlparse
from pathlib import Path
from CNNClassifier.entity.config_entity import EvaluationConfig
from CNNClassifier.utils.common import read_yaml, create_directories, save_json


class Evaluation:
    def __init__(self, config):
        self.config = config

    def _valid_generator(self):
        datagenerator_kwargs = dict(
            rescale=1. / 255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(**datagenerator_kwargs)

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        # Assuming save_json is a function you've defined elsewhere
        save_json(path=Path("scores.json"), data=scores)

    def generate_signature(self):
        # Define the input schema with correct dimensions
        input_schema = Schema([
            TensorSpec(np.dtype(np.float32), [-1, 224, 224, 3]),
        ])

        # Define the output schema with correct dimensions
        output_schema = Schema([
            TensorSpec(np.dtype(np.float32), [-1, 2]),
        ])

        return ModelSignature(inputs=input_schema, outputs=output_schema)

    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics({"loss": self.score[0], "accuracy": self.score[1]})

            signature = self.generate_signature()

            if tracking_url_type_store != "file":
                mlflow.keras.log_model(self.model, "model", signature=signature, registered_model_name="VGG16Model")
            else:
                mlflow.keras.log_model(self.model, "model", signature=signature)