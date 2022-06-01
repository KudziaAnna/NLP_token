Segmentacja oraz tokenizacja tekstu za pomocą sieci neuronowych dla języka angielskiego i włoskiego. Projekt realizowany w ramach przedmiotu Przetwarzanie języka naturalnego na Politechnice Warszawskiej
-----

Cechy
-----

* Organizacja kodu za pomocą `PyTorch Lightning <https://pytorch-lightning.readthedocs.io/en/latest/>`_.
* Wczytywanie danych za pomocą `datamodules` z `PyTorch Lightning Bolts <https://lightning-bolts.readthedocs.io/en/latest/>`_.
* Konfiguracja eksperymentów za pomocą pakietu `Hydra <https://hydra.cc/docs/next/tutorials/intro/>`_.
* Wizualizacja metryk za pomocą `Weights & Biases (wandb.ai) <https://docs.wandb.ai/>`_.
* Możliwość przechowywania checkpointów jako artefakty w wandb.ai.
* Formatowanie logów/informacji w konsoli za pomocą pakietu `rich <https://github.com/willmcgugan/rich>`_.
* Adnotacje typowania dla większości kodu.
* Zgodność z Python 3.9.

Instalacja
----------

Po sklonowaniu repozytorium tworzenie środowiska conda::

    $ conda env create -f environment.yml
    $ conda activate nlp_token

Konfiguracja środowiska w pliku ``.env`` po utworzeniu konta na wandb.ai::

    DATA_DIR=datasets
    RESULTS_DIR=results
    WANDB_ENTITY=WANDB_LOGIN
    WANDB_PROJECT=WANDB_PROJECT_NAME

``DATA_DIR`` jest katalogiem nadrzędnym do przechowywania zbiorów danych.
``RESULTS_DIR`` jest katalogiem nadrzędnym dla katalogów roboczych.


Uruchamianie eksperymentów
--------------------------

Uruchomienie uczenia z katalogu głównego (plik ``.env`` musi znaleźć się na ścieżce wyszukiwania)::

    $ python -m nlp_token.main experiment=gru_model_eng

Dodanie metadanych dla danego uruchomienia::
    
    $ python -m nlp_token.main experiment=gru_model_eng pl.max_epochs=150 experiment.batch_size=64

Testowanie działania biblioteki Stanza na danych:

    $ python stanza_tokenize/tokenize_stanza.py en


