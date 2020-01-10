import sys
from collections import defaultdict
from collections import deque
import torch

CURSOR_UP_ONE = '\x1b[1A' 
ERASE_LINE = '\x1b[2K'

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        self.series = []
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.series.append(value)
        self.count += 1
        self.total += value

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

class Logger(object):
    def __init__(self):
        self.log_info = {}
        self.iter = -1
        self.epoch = None
        self.init_print = True
        self.multiprocessing_cnt = 1
        self.last_iter = 0
        self.last_cnt = self.multiprocessing_cnt
        self.interval = 20

    def add(self, name):
        self.log_info[name] = SmoothedValue(self.interval)

    def get(self, name):
        return self.log_info[name]

    def set_multiprocessing_cnt(self, v):
        self.multiprocessing_cnt = v

    def set_interval(self, interval):
        self.interval = interval

    def log(self, iteration, **kwargs):

        if iteration > self.iter:
            self.iter = iteration
        else:
            print('iteration should get lager.')
            raise ValueError

        for k, v in kwargs.items():
            if k in self.log_info:
                self.log_info[k].update(v)
            else:
                print('unknow log key word "{}". pls add first.'.format(k))
                raise ValueError

    def wait(self, iteration, flush=False):
        
        self.print_log(flash_back=self.iter-iteration, flush=flush)
        return

    def print_log(self, flash_back=0, flush=False):
        if len(self.log_info) <= 0:
            return
        if not flush:
            sys.stdout.write('#[iter {:<5d}] '.format(self.iter-flash_back))
            sys.stdout.write(str({k:[v.avg, v.median] for k,v in self.log_info.items()}))
        else:
            s = '\n+' + '-' * 62 + '+\n|' + 'iter {:<57d}'.format(self.iter) + '|\n' + '+' + '-' * 62 + '+\n'
            line = ['', '', '', '', '', '']
            line_num = 3
            for i, (k,v) in enumerate(self.log_info.items()):
                idx = i % 3
                line[idx], line[idx+3] = str(k)[:20], str(v[-1])[:20]
                if idx == 2 or idx == len(self.log_info.items())-1:
                    s += '|{:<20s}|{:<20s}|{:<20s}|\n|{:<20s}|{:<20s}|{:<20s}|\n'.format(*line)
                    line = ['', '', '', '', '', '']
                    line_num += 2

            s += '+' + '-' * 62 + '+\n'
            line_num += 2
            if self.init_print:
                self.init_print = False
            else:
                s = (CURSOR_UP_ONE + ERASE_LINE) * line_num + s
            sys.stdout.write(s)
            sys.stdout.flush()


if __name__=='__main__':
    import time
    logger = Logger()
    logger.add('a')
    #logger.add('b')
    #logger.add('c')
    #logger.add('d')
    for i in range(100):
        #logger.log(i, a=i,b=i+1,c=i+2,d=i+3)
        logger.log(i, a=i)
        logger.print_log(flush=True)
        time.sleep(0.1)


        
