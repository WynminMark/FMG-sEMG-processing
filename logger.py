import logging
import os
import datetime
import sys

def logger_init(log_file_name ='monitor',
                log_level = logging.INFO,
                log_dir = './logs/',
                only_file = True):
    """
    logging 初始化函数
    
    Args:
    ------
    """
    
    # 指定路径
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_path = os.path.join(log_dir, log_file_name + '_' + str(datetime.datetime.now())[:10] + '.txt')
    formatter = '[%(asctime)s] - %(levelname)s: %(message)s'
    if only_file:
        logging.basicConfig(filename = log_path,
                            level = log_level,
                            format = formatter,
                            datefmt='%Y-%d-%m %H:%M:%S')
    else:
        logging.basicConfig(level = log_level,
                            format = formatter,
                            datefmt = '%Y-%d-%m %H:%M:%S',
                            handlers = [logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)])
        pass
    pass

if __name__ == '__main__':
    logger_init()

