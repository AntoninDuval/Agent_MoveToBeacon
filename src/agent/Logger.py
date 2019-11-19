import logging
import datetime

class Logger():
    def __init__(self):
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] - %(message)s',
            filename='../log/MTB_{}{}{}{}.txt'.format(datetime.datetime.now().hour,
                                                      datetime.datetime.now().day,
                                                      datetime.datetime.now().month,
                                                      datetime.datetime.now().year))
        self.logger = logging.getLogger()  # get the root logger

    def get_logger(self):
        return self.logger