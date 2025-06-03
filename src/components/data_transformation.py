import os
import sys
import re
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFECV

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    processed_train_path: str = os.path.join('artifacts', 'train_processed.csv')
    processed_test_path: str = os.path.join('artifacts', 'test_processed.csv')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def extract_bairro(self, endereco):
        """
        Extracts the neighborhood name from the address
        """
        match = re.search(r" -\s*([^,]+),", endereco)
        if match:
            return match.group(1).strip()
        else:
            match = re.search(r"^([^,]+),", endereco)
            if match:
                return match.group(1).strip()
            else:
                return None

    def get_data_transformer_object(self, df):
        """
        This function is responsible for data transformation
        """
        try:           
            logging.info('Padronizando o nome das colunas')

            df.rename(columns={
                               'condominio': 'nrCondominio',
                               'iptu': 'nrIptu',
                               'floorSize': 'nrArea(m2)',
                               'numberOfRooms': 'nrQuartos',
                               'numberOfBathroomsTotal': 'nrBanheiros',
                               'numberOfParkingSpaces': 'nrVagas',
                               'floorLevel': 'nrAndar',
                               'numberOfSuites': 'nrSuites',
                              }, inplace=True)

            logging.info('Padronização concluida')

            logging.info('Tratando valores duplicados e ausentes')

            df.drop_duplicates(subset='codigo_zapimoveis', inplace=True)
            df.fillna(0, inplace=True)

            logging.info('Tratamento concluido')

            logging.info('Limpando as variáveis quantitativas')

            int_cols = ['nrPreco',
                        'nrCondominio',
                        'nrIptu',
                        'nrArea(m2)',
                        'nrQuartos',
                        'nrBanheiros',
                        'nrVagas',
                        'nrAndar',
                        'nrSuites']

            for col in int_cols:
                df[col] = df[col].replace(r'[^\d]', '', regex=True)
                df[col] = df[col].replace('', '0')
                df[col] = df[col].fillna(0)
                df[col] = df[col].astype(int)

            logging.info('Limpeza concluida')

            logging.info('Tratamento de outliers')

            # Coluna nrPreco
            df['nrPreco'] = df['nrPreco'].replace({
                                                   3500: 350000,
                                                   120000000: 1200000,
                                                   399000000: 399000
                                                  })

            # Coluna nrCondominio
            df.loc[df['nrCondominio'] > 10000, 'nrCondominio'] = df.loc[df['nrCondominio'] > 10000, 'nrCondominio'] / 1000

            # Coluna nrIptu
            df['nrIptu'] = df['nrIptu'].replace({
                                                 400000: 400,
                                                 120000: 120,
                                                 17000: 170
                                                })

            # Coluna nrAndar
            df.loc[df['nrAndar'] > 25, 'nrAndar'] = df.loc[df['nrAndar'] > 25, 'nrAndar'] // 10

            # Coluna nrArea(m2)
            df['nrArea(m2)'] = df['nrArea(m2)'].replace({
                                                         11: 110
                                                        })

            logging.info('Tratamento concluido')

            logging.info('Realizando o feature engineering')

            df['bairro'] = df['endereco'].apply(self.extract_bairro)
            df['bairro'] = df['bairro'].replace('ap 21 - Gonzaga', 'Gonzaga')

            amenities_cat = df.iloc[:, 14:].drop('nrSuites', axis=1).columns
            dummies_amenities = pd.get_dummies(df[amenities_cat], prefix='d', prefix_sep='', drop_first=True, dtype=int)
            df = pd.concat([df.iloc[:, :14], df['nrSuites'], dummies_amenities], axis=1)

            df['dCondominio'] = np.where(df['nrCondominio'] > 0, 1, 0)
            df['dIptu'] = np.where(df['nrIptu'] > 0, 1, 0)
            df['dQuartos'] = np.where(df['nrQuartos'] > 0, 1, 0)
            df['dVagas'] = np.where(df['nrVagas'] > 0, 1, 0)
            df['dAndar'] = np.where(df['nrAndar'] > 0, 1, 0)
            df['dSuites'] = np.where(df['nrSuites'] > 0, 1, 0)

            df['nrQtdComodidades'] = df.iloc[:, 15:].sum(axis=1)
            df['nrPrecoFixo'] = df['nrCondominio'] + df['nrIptu']
            df['nrPrecoFixo_m2'] = df['nrPrecoFixo']/df['nrArea(m2)']
            df['nrIptu_m2'] = df['nrIptu']/df['nrArea(m2)']
            df['nrCondominio_m2'] = df['nrCondominio']/df['nrArea(m2)']
            df['nrComodos'] = df['nrBanheiros'] + df['nrQuartos'] + df['nrSuites']
            df['nrComodos_m2'] = df['nrComodos']/df['nrArea(m2)']

            df['nrBanheiros_Suites'] = np.where(df['nrSuites'] >= 1,
                                                df['nrBanheiros']/df['nrSuites'],
                                                df['nrBanheiros'])
            df['nrVagas_Quartos'] = np.where(df['nrQuartos'] >= 1,
                                            df['nrVagas']/df['nrQuartos'],
                                            df['nrVagas'])                                
            df['nrCondominio_Andar'] = np.where(df['nrAndar'] >= 1,
                                                df['nrCondominio'] / df['nrAndar'],
                                                df['nrCondominio'])

            df['dtCriacao'] = pd.to_datetime(df['dtCriacao'], dayfirst=True)
            df['dtCriacao_Mes'] = df['dtCriacao'].dt.month
            df['dtCriacao_Trimestre'] = df['dtCriacao'].dt.quarter
            df['dtCriacao_SemanaAno'] = df['dtCriacao'].dt.isocalendar().week
            df.fillna(0, inplace=True)

            logging.info('Feature engineering concluido')

            return df

        except Exception as e:
            raise CustomException(error_message=e, error_detail=sys)

    def initiate_data_transformation(self, raw_path):
        try:
            logging.info('Iniciando a transformação de dados')
            print('Iniciando a transformação de dados')

            data = pd.read_csv(raw_path, dtype=str)

            logging.info('Leitura dos dados completos')

            logging.info('Iniciando transformação do conjunto de dados')

            df_transform = self.get_data_transformer_object(data)

            logging.info('Separando os conjuntos de treino e teste')

            train_df, test_df = train_test_split(df_transform, test_size=0.3, random_state=42)

            train_df.to_csv(self.data_transformation_config.train_data_path, index=False, header=True)
            test_df.to_csv(self.data_transformation_config.test_data_path, index=False, header=True)

            logging.info('Separação dos conjuntos concluida')

            logging.info('Transformação dos dados concluida')
            print('Transformação dos dados concluida')
            
            return train_df, test_df

        except Exception as e:
            raise CustomException(error_message=e, error_detail=sys)

    def feature_selection(self):
        try:
            logging.info('Selecionando as melhores features')
            print('Selecionando as melhores features')

            train_df = pd.read_csv(self.data_transformation_config.train_data_path)
            test_df = pd.read_csv(self.data_transformation_config.test_data_path)

            y_train = train_df['nrPreco']
            X_train = train_df.drop(['nrPreco', 'codigo_zapimoveis', 'dtCriacao', 'codigo_anunciante', 'imobiliaria', 'dtColeta', 'endereco'], axis=1)

            y_test = test_df['nrPreco']
            X_test = test_df.drop(['nrPreco', 'codigo_zapimoveis', 'dtCriacao', 'codigo_anunciante', 'imobiliaria', 'dtColeta', 'endereco'], axis=1)
            
            regressor = LinearRegression()
            regressor.fit(X_train, y_train)

            rfe = RFECV(regressor, step=1, cv=5, scoring='neg_root_mean_squared_error')
            X_train = rfe.fit_transform(X_train, y_train)
            X_test = rfe.transform(X_test)

            rfe_columns = rfe.get_feature_names_out()
            X_train = pd.DataFrame(X_train, columns=rfe_columns)
            X_test = pd.DataFrame(X_test, columns=rfe_columns)

            logging.info('Seleção de melhores features concluida')

            processed_train_df = pd.concat([X_train, y_train], axis=1)
            processed_test_df = pd.concat([X_test, y_test], axis=1)

            processed_train_df.to_csv(self.data_transformation_config.processed_train_path, index=False, header=True)
            processed_test_df.to_csv(self.data_transformation_config.processed_test_path, index=False, header=True)

            logging.info('Salvamento dos dados concluido')
            print('Seleção de melhores features concluida')

            return self.data_transformation_config.processed_train_path, self.data_transformation_config.processed_test_path
        
        except Exception as e:
            raise CustomException(error_message=e, error_detail=sys)