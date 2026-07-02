import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename='./self-test-improvement/Python高级/log/log.txt',
                    filemode='w')

test_logger = logging.getLogger('test_logger')

file_handler = logging.FileHandler('./self-test-improvement/Python高级/log/test_logger.txt', mode='w')
test_logger.addHandler(file_handler)

# logging.getLogger(__name__)
# logging.debug('debug msg')
# logging.info('info msg')
# logging.warning('warn msg')
# logging.error('error msg')
# logging.critical('critical msg')

test_logger.error('error from custom logger')

