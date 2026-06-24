import logging
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import Queue

def setup_worker_logging(log_queue):
    """Configures workers to send logs to the central queue."""
    h = QueueHandler(log_queue)
    root = logging.getLogger()
    root.addHandler(h)
    root.setLevel(logging.DEBUG)

def setup_master_logging():
    """Configures the main process to listen to the queue and write to a file."""
    log_queue = Queue(-1)
    
    # Configure file output
    file_handler = logging.FileHandler("simulation_debug.log", mode="w")
    formatter = logging.Formatter('%(asctime)s | %(processName)s | %(levelname)s | %(message)s')
    file_handler.setFormatter(formatter)
    
    listener = QueueListener(log_queue, file_handler)
    listener.start()
    return listener, log_queue