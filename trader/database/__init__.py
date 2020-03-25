import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vnpy.trader.database.database import BaseDatabaseManager

if "VNPY_TESTING" not in os.environ:
    from vnpy_of_python3.examples.vnpy.trader.setting import get_settings
    from .initialize import init

    settings = get_settings("database.")
    mongo_manager: "BaseDatabaseManager" = init(settings=settings)