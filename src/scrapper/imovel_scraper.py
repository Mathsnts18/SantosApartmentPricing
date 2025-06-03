import sys
import os
import csv
import json
import re
from datetime import date
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from driver import Driver

from tqdm import tqdm
from selenium.webdriver.common.by import By
from selenium.webdriver import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

@dataclass
class ImoveisConfig:
    imoveis_path: str = os.path.join('data/raw', 'imoveis.json')
    filtered_links_path: str = os.path.join('artifacts', 'filtered_links.csv')

class ImovelScraper:
    def __init__(self):
        self.imoveis_config = ImoveisConfig()
        self.imoveis = set()
        self.links = set()
        self.filtered_links = list()

    def load_links(self):
        logging.info("Carregando links existentes...")

        try:
            if os.path.exists('artifacts/links_imovel.csv'):
                with open('artifacts/links_imovel.csv', 'r') as arquivo_existente:
                    reader = csv.reader(arquivo_existente)
                    for linha in reader:
                        if linha:
                            self.links.add(linha[0])
                logging.info(f"Links carregados com sucesso.")
                print(f"Links carregados com sucesso! Total de links: {len(self.links)}")
            else:
                logging.info("Nenhum link carregado.")
                print("Nenhum link carregado.")

        except Exception as e:
            raise CustomException(error_message=e, error_detail=sys)

    def filter_links(self):
        self.load_links()

        logging.info("Filtrando links...")
        try:
            self.filtered_links = [link.strip() for link in self.links if '/lancamento/' not in link]

            os.makedirs(name=os.path.dirname(p=self.imoveis_config.filtered_links_path), exist_ok=True)
            with open("artifacts/filtered_links.csv", "a", newline="") as arquivo:
                writer = csv.writer(arquivo)
                writer.writerows([link] for link in self.filtered_links)
            logging.info(f"Links filtrados com sucesso.")
            print(f"Links filtrados com sucesso! Total de links: {len(self.filtered_links)}")
        except Exception as e:
            raise CustomException(error_message=e, error_detail=sys)

    def load_imoveis(self):
        logging.info("Carregando imóveis existentes...")
        try:
            os.makedirs(name=os.path.dirname(p=self.imoveis_config.imoveis_path), exist_ok=True)
            if os.path.exists('data/raw/imoveis.json'):
                with open('data/raw/imoveis.json', 'r') as arquivo_existente:
                    for linha in arquivo_existente:
                        if linha.strip():
                            dado = json.loads(linha)
                            self.imoveis.add(dado['codigo_zapimoveis'])
                logging.info(f"Imóveis carregados com sucesso.")
                print(f"Imóveis carregados com sucesso! Total de imóveis: {len(self.imoveis)}")
            else:
                logging.info("Nenhum imóvel carregado.")
                print("Nenhum imóvel carregado.")

        except Exception as e:
            raise CustomException(error_message=e, error_detail=sys)

        
    def scraper(self):
        logging.info("Iniciando o scrapping...")

        try:
            self.load_imoveis()
            self.filter_links()

            pbar = tqdm(self.filtered_links, desc="Scrapping imóveis")
            for link in pbar:
                driver = Driver().initialize_driver()
                driver.get(url=link)

                try:
                    WebDriverWait(driver, 10).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, '[data-testid="price-info-value"]'))
                        )
                except:
                    print(f"\nErro 404 encontrado na página {link}. Pulando para a próxima...\n")
                    driver.close()
                    # Se o elemento de erro 404 não for encontrado, continua normalmente
                    continue

                imovel_info = {}

                # Código do anunciante
                advertiser_code = driver.find_element(By.CSS_SELECTOR, value='[data-cy="ldp-propertyCodes-txt"]').text
                advertiser_code = re.search(r":\s(.+)\s\|", advertiser_code).group()[1:-1].strip()

                pbar.set_description(f"Coletando ID {advertiser_code}...")
                pbar.set_postfix(imoveis_coletados=len(self.imoveis))

                if advertiser_code and advertiser_code not in self.imoveis:
                    # Código do anunciante
                    imovel_info['codigo_anunciante'] = advertiser_code

                    # ID do imóvel
                    id_number = re.search(r"id-(\d+)", link)
                    imovel_info['codigo_zapimoveis'] = id_number.group(1)

                    # Data de criação
                    meses = {
                            'janeiro': '01', 'fevereiro': '02', 'março': '03', 'abril': '04',
                            'maio': '05', 'junho': '06', 'julho': '07', 'agosto': '08',
                            'setembro': '09', 'outubro': '10', 'novembro': '11', 'dezembro': '12'
                            }
                
                    created_at = driver.find_element(By.CSS_SELECTOR, value='[data-testid="listing-created-date"]').text
                    created_at = re.search(r"criado em (\d{1,2}) de (\w+) de (\d{4})", created_at)
                    dia, mes_str, ano = created_at.groups()
                    mes_num = meses[mes_str.lower()]
                    imovel_info['dtCriacao'] = f"{int(dia):02d}/{mes_num}/{ano}"

                    # Código do anunciante
                    advertiser_code = driver.find_element(By.CSS_SELECTOR, value='[data-cy="ldp-propertyCodes-txt"]').text
                    imovel_info['codigo_anunciante'] = re.search(r":\s(.+)\s\|", advertiser_code).group()[1:-1].strip()

                    # Data de coleta
                    imovel_info['dtColeta'] = date.today().strftime("%d/%m/%Y")
                    # Imobiliaria
                    imovel_info['imobiliaria'] = driver.find_elements(By.CSS_SELECTOR, value='[data-testid="official-store-redirect-link"]')[1].text

                    # Preço do imóvel
                    imovel_info['nrPreco'] = driver.find_element(By.CSS_SELECTOR, value='[data-testid="price-info-value"]').text
        
                    # Preço do condomínio
                    imovel_info['condominio'] = driver.find_element(By.CSS_SELECTOR, value='[id="condo-fee-price"]').text
        
                    # IPTU
                    imovel_info['iptu'] = driver.find_element(By.CSS_SELECTOR, value='[id="iptu-price"]').text
        
                    # Endereço
                    imovel_info['endereco'] = driver.find_element(By.CSS_SELECTOR, value='[data-testid="address-info-value"]').text

                    # Comodidades
                    try:
                        driver.find_element(By.CSS_SELECTOR, '[data-cy="ldp-TextCollapse-btn"]').click()
                    except:
                        pass
                    
                    amenities = driver.find_elements(By.CSS_SELECTOR, '[class="amenities-item"]')

                    for amenity in amenities:
                        var_amenity = amenity.get_attribute("itemprop")
                        value = amenity.text
                        imovel_info[var_amenity] = value

                    # Salva o imovel_info no arquivo JSON
                    with open('data/raw/imoveis.json', 'a') as arquivo:
                        json.dump(imovel_info, arquivo, ensure_ascii=False)
                        arquivo.write('\n')
                    self.imoveis.add(advertiser_code)

                driver.close()
            
            logging.info("Dados exportados para imoveis.json com sucesso.")
            print(f"Imoveis carregados com sucesso! Total de imoveis: {len(self.imoveis)}")

        except Exception as e:
            raise CustomException(error_message=e, error_detail=sys)


if __name__ == "__main__":
    imovel_scraper = ImovelScraper()
    imovel_scraper.scraper()