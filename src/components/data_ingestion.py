import os
import sys
from dataclasses import dataclass

import pandas as pd

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('data/raw', 'data.csv')


class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Iniciando a ingestão de dados')
        print('Iniciando a ingestão de dados')

        try:
            df = pd.read_json('data/raw/imoveis.json', lines=True)
            logging.info('Leitura dos dados concluída')

            os.makedirs(
                name=os.path.dirname(p=self.ingestion_config.raw_data_path),
                exist_ok=True,
            )
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            logging.info('Salvamento dos dados concluído')

            logging.info('Ingestão de dados concluida')
            print('Ingestão de dados concluida')

            return self.ingestion_config.raw_data_path

        except Exception as e:
            raise CustomException(error_message=e, error_detail=sys)


if __name__ == '__main__':
    obj = DataIngestion()
    raw_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_df, test_df = data_transformation.initiate_data_transformation(raw_data)

    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(train_df, test_df)
    model_trainer.save_model()