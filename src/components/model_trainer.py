import json
import os
import sys
from dataclasses import dataclass

import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    model_file_path: str = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_data_path, test_data_path):
        logging.info('Iniciando o treinamento do modelo')
        print('Iniciando o treinamento do modelo')

        try:
            train_df = pd.read_csv(train_data_path)

            X_train = train_df.drop('nrPreco', axis=1)
            y_train = train_df['nrPreco']

            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', CatBoostRegressor(verbose=0, random_state=42)),
            ])
            pipe.fit(X_train, y_train)

            cv_rmse = cross_val_score(
                pipe,
                X_train,
                y_train,
                cv=5,
                scoring='neg_root_mean_squared_error',
            )
            print(
                f'Média do RMSE da validação cruzada: {round(-cv_rmse.mean(), 2)}'
            )

            test_df = pd.read_csv(test_data_path)

            X_test = test_df.drop('nrPreco', axis=1)
            y_test = test_df['nrPreco']

            save_object(
                file_path=self.model_trainer_config.model_file_path, obj=pipe
            )

            logging.info('Modelo treinado com sucesso')
            print('Modelo treinado com sucesso')

            metrics = {
                'mean_absolute_error_X_test': mean_absolute_error(
                    y_true=y_test, y_pred=pipe.predict(X_test)
                ),
                'r2_score_X_test': r2_score(
                    y_true=y_test, y_pred=pipe.predict(X_test)
                ),
                'root_mean_squared_error_X_test': root_mean_squared_error(
                    y_true=y_test, y_pred=pipe.predict(X_test)
                ),
            }

            with open('artifacts/metrics.txt', 'w') as f:
                json.dump(metrics, f, indent=4)

            logging.info('Métricas do modelo salvas com sucesso')
            print('Métricas do modelo salvas com sucesso')

            return self.model_trainer_config.model_file_path

        except Exception as e:
            raise CustomException(error_message=e, error_detail=sys)
