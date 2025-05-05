# src/utils/logger.py

import logging


def setup_logger(name: str, level: str = 'INFO'):
    """
    Setup di un logger che stampa i log sulla console.

    Args:
    - name: Nome del logger.
    - level: Il livello di log da settare. Default è 'INFO'.

    Returns:
    - logger: Un oggetto logger configurato.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Crea il formato dei log
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Crea un handler per la console (stampa sulla console)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Aggiungi il handler al logger
    logger.addHandler(console_handler)

    return logger
