import time

from colorama import Style, Fore


class Logger:
    # Here will be the instance stored.
    __instance = None

    @staticmethod
    def getInstance():
        """ Static access method. """
        if Logger.__instance == None:
            Logger()
        return Logger.__instance

    def __init__(self):
        """ Virtually private constructor. """
        if Logger.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            Logger.__instance = self
        self.time_origin = time.time()
        self.filename = None
        self.LEVELS = {'DEBUG': 0, 'INFO': 1, 'WARN': 2, 'ERROR': 3, 'FATAL': 4}
        self.COLORS = {'DEBUG': Style.DIM, 'INFO': Fore.GREEN, 'WARN': Fore.YELLOW, 'ERROR': Fore.RED,
                       'FATAL': Fore.RED + Style.BRIGHT}
        self.log_level = 1

    def set_logfile(self, filename):
        self.filename = filename

    def set_log_level(self, log_level):
        self.log_level = log_level

    def _log(self, *args, sep=' ', end='\n', flush=False, level='INFO'):
        if self.LEVELS[level] >= self.log_level:
            elapsed_time = '[%12.5fs, %5s]' % (time.time() - self.time_origin, level)
            print(self.COLORS[level], elapsed_time, *args, Style.RESET_ALL, sep=sep, end=end, flush=flush)
            if self.filename:
                f = open(self.filename, 'a')
                print(elapsed_time, *args, sep=sep, end=end, flush=flush, file=f)
                f.close()

    def __call__(self, *args, sep=' ', end='\n', flush=False, level='INFO'):
        self._log(*args, sep=sep, end=end, flush=flush, level=level)

    def debug(self, *args, sep=' ', end='\n', flush=False):
        self._log(*args, sep=sep, end=end, flush=flush, level='DEBUG')

    def info(self, *args, sep=' ', end='\n', flush=False):
        self._log(*args, sep=sep, end=end, flush=flush, level='INFO')

    def warn(self, *args, sep=' ', end='\n', flush=False):
        self._log(*args, sep=sep, end=end, flush=flush, level='WARN')

    def error(self, *args, sep=' ', end='\n', flush=False):
        self._log(*args, sep=sep, end=end, flush=flush, level='ERROR')

    def fatal(self, *args, sep=' ', end='\n', flush=False):
        self._log(*args, sep=sep, end=end, flush=flush, level='FATAL')


glogger = Logger.getInstance()
glogger.set_log_level(0)
