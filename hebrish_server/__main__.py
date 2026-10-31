import logging

import uvicorn

from . import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
uvicorn.run("hebrish_server.app:app", host=config.HOST, port=config.PORT, log_level="info")
