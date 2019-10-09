if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        filename='filename.txt')
    logger = logging.getLogger()  # get the root logger
    logger.warning('This should go in the file.')
    print(logger.handlers)