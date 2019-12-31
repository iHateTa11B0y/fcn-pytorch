import sys
CURSOR_UP_ONE = '\x1b[1A' 
ERASE_LINE = '\x1b[2K'

class Logger(object):
    def __init__(self):
        self.log_info = {}
        self.iter = -1
        self.epoch = None
        self.init_print = True
        self.multiprocessing_cnt = 1
        self.last_iter = 0
        self.last_cnt = self.multiprocessing_cnt

    def add(self, name):
        self.log_info[name] = []

    def get(self, name):
        return self.log_info[name]

    def set_multiprocessing_cnt(self, v):
        self.multiprocessing_cnt = v

    def log(self, iteration, **kwargs):

        if iteration > self.iter:
            self.iter = iteration
        else:
            print('iteration should get lager.')
            raise ValueError

        for k, v in kwargs.items():
            if k in self.log_info:
                self.log_info[k].append(v)
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
            print('#[iter {:<5d}] '.format(self.iter-flash_back))
            print({k:v[-1-flash_back] for k,v in self.log_info.items()})
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


        
