import logging
import os
from datetime import datetime

now = datetime.now()
log_filename = f'{now.strftime("%H_%M_%S")}.log'
log_folder = now.strftime("%Y/%m/%d")

# Criar diretório para armazenar os logs
logs_path = os.path.join(os.getcwd(), 'logs', log_folder)
os.makedirs(logs_path, exist_ok=True)

# Caminho do arquivo de log
log_file_path = os.path.join(logs_path, log_filename)

# Configurar logging
logging.basicConfig(
    filename=log_file_path,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)