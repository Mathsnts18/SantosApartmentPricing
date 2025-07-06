import os
import re
import sys
from dataclasses import dataclass

import pandas as pd
from feature_engine.outliers import Winsorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join(
        'artifacts', 'preprocessor.pkl'
    )
    train_data_path: str = os.path.join('data/processed', 'train.csv')
    test_data_path: str = os.path.join('data/processed', 'test.csv')
    processed_train_path: str = os.path.join(
        'data/processed', 'train_processed.csv'
    )
    processed_test_path: str = os.path.join(
        'data/processed', 'test_processed.csv'
    )


class DataProcessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        pd.set_option('future.no_silent_downcasting', True)

        remove_columns = [
            'codigo_zapimoveis',
            'dtCriacao',
            'codigo_anunciante',
            'imobiliaria',
            'dtColeta',
        ]

        X = X.drop(remove_columns, axis=1)

        X.rename(
            columns={
                'condominio': 'nrCondominio',
                'iptu': 'nrIptu',
                'floorSize': 'nrArea(m2)',
                'numberOfRooms': 'nrQuartos',
                'numberOfBathroomsTotal': 'nrBanheiros',
                'numberOfParkingSpaces': 'nrVagas',
                'floorLevel': 'nrAndar',
                'numberOfSuites': 'nrSuites',
            },
            inplace=True,
        )

        X.fillna(0, inplace=True)
        X.replace('', 0, inplace=True)

        int_cols = [
            'nrPreco',
            'nrCondominio',
            'nrIptu',
            'nrArea(m2)',
            'nrQuartos',
            'nrBanheiros',
            'nrVagas',
            'nrAndar',
            'nrSuites',
        ]

        for col in int_cols:
            X[col] = X[col].replace(r'[^\d]', '', regex=True)
            X[col] = X[col].replace('', '0')
            X[col] = X[col].fillna(0)
            X[col] = X[col].astype(int)

        endereco_series = X['endereco']

        def extract_bairro(endereco):
            match = re.search(r' -\s*([^,]+),', endereco)
            if match:
                return match.group(1).strip()
            else:
                match = re.search(r'^([^,]+),', endereco)
                if match:
                    return match.group(1).strip()
                else:
                    return None

        X['bairro'] = endereco_series.apply(extract_bairro)
        X['bairro'] = X['bairro'].replace({
            'ap 21 - Gonzaga': 'Gonzaga',
            'ap 22 - Boqueirão': 'Boqueirão',
            'apto 127 - Marapé': 'Marapé',
            '57 - Boqueirão': 'Boqueirão',
            '91 - Campo Grande': 'Campo Grande',
        })
        X.drop(columns=['endereco'], inplace=True)

        X.drop_duplicates(inplace=True)

        return X.reset_index(drop=True)


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self, raw_path):
        try:
            logging.info('Iniciando a transformação de dados')
            print('Iniciando a transformação de dados')

            data = pd.read_csv(raw_path, dtype=str)

            logging.info('Leitura dos dados completos')

            logging.info('Iniciando o processamento de dados')

            processor = DataProcessor()
            abt = processor.fit_transform(data)

            logging.info('Separando os conjuntos de treino e teste')

            train_df, test_df = train_test_split(
                abt, test_size=0.3, random_state=42
            )

            logging.info(
                'Aplicando o Winsorizer no método MAD nos conjuntos de treino e teste'
            )

            winsor = Winsorizer(
                capping_method='mad',
                tail='both',
                variables=['nrPreco', 'nrCondominio', 'nrIptu', 'nrArea(m2)'],
            )

            train_df = winsor.fit_transform(train_df)
            test_df = winsor.transform(test_df)

            logging.info('Winsorizer aplicado com sucesso')

            train_df.to_csv(
                self.data_transformation_config.train_data_path,
                index=False,
                header=True,
            )
            test_df.to_csv(
                self.data_transformation_config.test_data_path,
                index=False,
                header=True,
            )

            logging.info('Separação dos conjuntos concluida')

            logging.info('Transformação dos dados concluida')
            print('Transformação dos dados concluida')

            return train_df, test_df

        except Exception as e:
            raise CustomException(error_message=e, error_detail=sys)
