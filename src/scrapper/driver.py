import sys

from src.logger import logging
from src.exception import CustomException

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

class Driver:
    def __init__(self):
        self.header = "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36"

    def set_properties(self):
        chrome_options = Options()
        chrome_options.add_argument(self.header)
        chrome_options.add_argument("--headless")

        return chrome_options

    def initialize_driver(self):
        try:
            chrome_options = self.set_properties()
            driver = webdriver.Chrome(options=chrome_options)
            logging.info("Driver inicializado com sucesso")
            return driver

        except Exception as e:
            raise CustomException(error_message=e, error_detail=sys)

    def get_url(url):
        try:
            driver.get(url=url)

        except Exception as e:
            raise CustomException(error_message=e, error_detail=sys)

    def close_driver(driver):
        try:
            return driver.close()

        except Exception as e:
            raise CustomException(error_message=e, error_detail=sys)

    def quit_driver(driver):
        try:
            return driver.quit()
            logging.info("Driver fechado com sucesso")

        except Exception as e:
            raise CustomException(error_message=e, error_detail=sys)