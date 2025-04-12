import logging

def filter_loggers(lib_level_dict: dict[str, str]):
    """
    Set the logging level for specific libraries.

    Args:
        lib_level_dict: Dictionary mapping library names to logging levels.
    """
    for lib_name, level in lib_level_dict.items():
        if level not in {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}:
            raise ValueError(f"Invalid logging level: {level} for library: {lib_name}")
        logging.getLogger(lib_name).setLevel(level)
