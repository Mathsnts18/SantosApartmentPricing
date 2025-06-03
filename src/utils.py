import sys
import os
import pickle
from src.exception import CustomException

def save_object(file_path, obj):
    """
    Save an object to a file using pickle.
    
    :param file_path: Path where the object will be saved.
    :param obj: The object to be saved.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
    except Exception as e:
        raise CustomException(error_message=e, error_detail=sys)