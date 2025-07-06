import json
import os
import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
    mean_absolute_percentage_error
)
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import RFECV
from feature_engine.encoding import OneHotEncoder
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object



@dataclass
class ModelTrainerConfig:
    model_file_path: str = os.path.join('artifacts', 'model.pkl')

class NewFeats(BaseEstimator, TransformerMixin):
    """
    Classe para criar novas features no DataFrame.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['dCondominio'] = np.where(X['nrCondominio'] > False, True, False)
        X['dIptu'] = np.where(X['nrIptu'] > False, True, False)
        X['dQuartos'] = np.where(X['nrQuartos'] > False, True, False)
        X['dVagas'] = np.where(X['nrVagas'] > False, True, False)
        X['dAndar'] = np.where(X['nrAndar'] > False, True, False)
        X['dSuites'] = np.where(X['nrSuites'] > False, True, False)

        X['nrQtdComodidades'] = X.iloc[:, 15:].sum(axis=1)
        X['nrPrecoFixo'] = X['nrCondominio'] + X['nrIptu']
        X['nrPrecoFixo_m2'] = X['nrPrecoFixo']/X['nrArea(m2)']
        X['nrIptu_m2'] = X['nrIptu']/X['nrArea(m2)']
        X['nrCondominio_m2'] = X['nrCondominio']/X['nrArea(m2)']
        X['nrComodos'] = X['nrBanheiros'] + X['nrQuartos'] + X['nrSuites']
        X['nrComodos_m2'] = X['nrComodos']/X['nrArea(m2)']

        X['nrBanheiros_Suites'] = np.where(X['nrSuites'] >= 1,
                                            X['nrBanheiros']/X['nrSuites'],
                                            X['nrBanheiros'])
        X['nrVagas_Quartos'] = np.where(X['nrQuartos'] >= 1,
                                        X['nrVagas']/X['nrQuartos'],
                                        X['nrVagas'])                                
        X['nrCondominio_Andar'] = np.where(X['nrAndar'] >= 1,
                                            X['nrCondominio'] / X['nrAndar'],
                                            X['nrCondominio'])

        X.fillna(0, inplace=True)
        return X
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_data, test_data):
        logging.info('Iniciando o treinamento do modelo')
        print('Iniciando o treinamento do modelo')

        try:
            train_df = train_data
            test_df = test_data
            logging.info('Dados de treino e teste carregados com sucesso')
            
            X_train = train_df.drop('nrPreco', axis=1)
            y_train = train_df['nrPreco']
            X_test = test_df.drop('nrPreco', axis=1)
            y_test = test_df['nrPreco']

            ohe = OneHotEncoder(drop_last_binary=True)
            new_feats = NewFeats()
            rfe = RFECV(
                estimator=DecisionTreeRegressor(random_state=42),
                step=1,
                cv=5,
                scoring='neg_root_mean_squared_error',
                verbose=0
            )
            scaler = StandardScaler()

            logging.info('Criando o pipeline de treinamento do modelo')
            pipe = Pipeline([
                ('One Hot Encoder', ohe),
                ('New Features', new_feats),
                ('RFE', rfe),
                ('scaler', scaler)
            ])

            X_train_transformed = pipe.fit_transform(X_train, y_train)
            X_test_transformed = pipe.transform(X_test)
            X_train_transformed = pd.DataFrame(X_train_transformed, columns=rfe.get_feature_names_out())
            X_test_transformed = pd.DataFrame(X_test_transformed, columns=rfe.get_feature_names_out())

            # Define a URL do servidor MLflow
            mlflow.set_tracking_uri('http://127.0.0.1:5000')

            # Define o experimento no qual os dados serão registrados
            mlflow.set_experiment(experiment_id=542105102691604280)

            with mlflow.start_run():

                model = CatBoostRegressor(random_state=42, verbose=0)

                model.fit(X_train_transformed, y_train)

                # Validação cruzada no conjunto de treino
                cv_results = cross_validate(model, X_train_transformed, y_train, cv=5, scoring=['neg_root_mean_squared_error', 'neg_mean_absolute_error', 'r2', 'neg_mean_absolute_percentage_error'])
                
                # Métricas no conjunto de teste
                y_pred = model.predict(X_test_transformed)
                root_mean_squared_error_test = root_mean_squared_error(y_test, y_pred)
                mean_absolute_error_test = mean_absolute_error(y_test, y_pred)
                r2_score_test = r2_score(y_test, y_pred)
                mean_absolute_percentage_error_test = mean_absolute_percentage_error(y_test, y_pred)

                mlflow.log_metrics({
                                'RMSE_CV': -cv_results['test_neg_root_mean_squared_error'].mean(),
                                'RMSE_test': root_mean_squared_error_test,
                                'RMSE_diff': abs(-cv_results['test_neg_root_mean_squared_error'].mean() - root_mean_squared_error_test),
                                'MAE_CV': -cv_results['test_neg_mean_absolute_error'].mean(),
                                'MAE_test': mean_absolute_error_test,
                                'R2_CV': -cv_results['test_r2'].mean(),
                                'R2_test': r2_score_test,
                                'MAPE_CV': -cv_results['test_neg_mean_absolute_percentage_error'].mean(),
                                'MAPE_test': mean_absolute_percentage_error_test
                                })
                
                # Hiperparâmetros do pré-processamentos
                mlflow.log_param('1. OneHotEncoder', ohe.get_params())
                mlflow.log_param('2. NewFeats', new_feats.get_params())
                mlflow.log_param('3. RFE', rfe.get_params())
                mlflow.log_param('4. StandardScaler', scaler.get_params())
                mlflow.log_param('5. Regressor', model.get_params())
                
                # Signature
                mlflow.sklearn.log_model(
                    model, "CatBoostRegressor", input_example=X_train_transformed.iloc[[0]]
                )

                # Gráfico real vs predito
                plt.figure(figsize=(10, 10))
                sns.scatterplot(x=y_test, y=y_pred)
                plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                plt.ylabel('Valor Previsto')
                plt.title('Comparação entre Valor Real e Valor Previsto')
                tmpdir = tempfile.mkdtemp()
                plot_path = os.path.join(tmpdir, "real_vs_previsto.png")
                plt.savefig(plot_path)
                mlflow.log_artifact(plot_path)

            logging.info('Modelo treinado com sucesso')

            return print('Modelo treinado com sucesso')

        except Exception as e:
            raise CustomException(error_message=e, error_detail=sys)
        
    def save_model(self):
        try:
            # Define a URL do servidor MLflow
            mlflow.set_tracking_uri('http://127.0.0.1:5000')

            # Define o experimento no qual os dados serão registrados
            mlflow.set_experiment(experiment_id=542105102691604280)
            
            client = mlflow.tracking.MlflowClient()
            version = max([int(i.version) for i in client.get_latest_versions('Price_Model')])

            if not version:
                    raise CustomException('Nenhuma versão de modelo salva, escolha um no MLflow')
            
            logging.info('Modelo escolhido')
            model = mlflow.sklearn.load_model(f'models:/Price_Model/{version}')

            save_object(
                file_path=self.model_trainer_config.model_file_path,
                obj=model
            )

            return f'Versão do modelo: {version}'
        except Exception as e:
            raise CustomException(error_message=e, error_detail=sys)
