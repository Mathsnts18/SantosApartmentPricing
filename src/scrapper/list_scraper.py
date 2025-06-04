import csv
import os
import sys
from dataclasses import dataclass

from driver import Driver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tqdm import tqdm

from src.exception import CustomException
from src.logger import logging


@dataclass
class LinksConfig:
    links_path: str = os.path.join('artifacts', 'links_imovel.csv')


class ListScraper:
    def __init__(self):
        self.links_config = LinksConfig()
        self.base_url = 'https://www.zapimoveis.com.br/venda/apartamentos/sp+santos/?transacao=venda&onde=,S%C3%A3o%20Paulo,Santos&tipos=apartamento_residencial&pagina='
        self.links = set()
        self.new_links = list()

    def load_links(self):
        logging.info('Carregando links existentes...')

        try:
            os.makedirs(
                name=os.path.dirname(p=self.links_config.links_path),
                exist_ok=True,
            )
            if os.path.exists('artifacts/links_imovel.csv'):
                with open(
                    'artifacts/links_imovel.csv', 'r'
                ) as arquivo_existente:
                    reader = csv.reader(arquivo_existente)
                    for linha in reader:
                        if linha:
                            self.links.add(linha[0])
                logging.info('Links carregados com sucesso.')
                print(
                    f'Links carregados com sucesso! Total de links: {len(self.links)}'
                )
            else:
                logging.info('Nenhum link carregado.')
                print('Nenhum link carregado.')

        except Exception as e:
            raise CustomException(error_message=e, error_detail=sys)

    def scrapper(self):
        logging.info('Iniciando o scrapping...')

        self.load_links()

        page = 1
        pbar = tqdm()

        while True:
            url = self.base_url + str(page)

            try:
                driver = Driver().initialize_driver()
                driver.get(url=url)

                card_imovel = driver.find_elements(
                    By.CSS_SELECTOR, '[data-cy="rp-property-cd"]'
                )

                try:
                    ActionChains(driver).move_to_element(
                        card_imovel[-1]
                    ).pause(4).move_to_element(card_imovel[-1]).perform()
                    WebDriverWait(driver, 10).until(
                        EC.visibility_of_all_elements_located((
                            By.CSS_SELECTOR,
                            '[data-cy="rp-property-cd"]',
                        ))
                    )
                except:
                    print('\nFim das páginas.')
                    logging.info('Fim das páginas.')
                    driver.close()
                    break

                links_found = 0
                new_links_found = 0
                for imovel in card_imovel:
                    link = imovel.find_element(By.TAG_NAME, 'a').get_attribute(
                        'href'
                    )

                    if link and link not in self.links:
                        with open(
                            'artifacts/links_imovel.csv', 'a', newline=''
                        ) as arquivo:
                            writer = csv.writer(arquivo)
                            writer.writerow([link])
                        self.links.add(link)
                        self.new_links.append(link)
                        new_links_found += 1
                        links_found += 1
                    else:
                        links_found += 1

                pbar.set_description(f'Página {page}')
                pbar.set_postfix(
                    links_encontrados=links_found, novos=len(self.new_links)
                )
                pbar.update(1)

                if links_found == 0:
                    print('Nenhum imóvel encontrado ou fim das páginas.')
                    logging.info(
                        'Nenhum imóvel encontrado ou fim das páginas.'
                    )
                    driver.close()
                    break

                driver.close()
                page += 1

            except Exception as e:
                raise CustomException(error_message=e, error_detail=sys)

        logging.info('Fim do scrapping.')
        return print(
            f'Foram encontrados {len(self.new_links)} novos links.\nTotal de links: {len(self.links)}'
        )


if __name__ == '__main__':
    list_scraper = ListScraper()
    list_scraper.scrapper()
