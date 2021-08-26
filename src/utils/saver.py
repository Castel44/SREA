import os
from datetime import datetime

import numpy as np


class Saver(object):
    # TODO: find better way to get dynamically OUTPATH without git
    def __init__(self, OUTPATH, name, hierarchy=''):
        self.out = OUTPATH
        self.name = name
        self.hierarchy = hierarchy

        self.f = None

        self.path = self.create_path_()
        self.makedir_()

    def create_path_(self):
        now = datetime.now()
        now = now.isoformat().split(sep='.')[0]
        return os.path.join(self.out, self.name, self.hierarchy, now)

    def makedir_(self):
        os.makedirs(self.path, exist_ok=True)

    def save_fig(self, fig, name='figure', **kwargs):
        filename = os.path.join(self.path, name)
        i = 0
        while os.path.exists('{}{:d}.png'.format(filename, i)):
            i += 1

        fig.savefig('{}{:d}.png'.format(filename, i), **kwargs)

    def get_path(self):
        return self.path

    def save_data(self, filename, x):
        np.save(os.path.join(self.path, filename), x, allow_pickle=True)

    def load_npdata(self, filename):
        return np.load(os.path.join(self.path, [filename + '.npy']))

    def init_log(self):
        print(f'LOGFILE created at {self.path}')
        self.f = open(os.path.join(self.path, 'LOG.txt'), 'w+')
        self.f.write('Script execution time: %s\r\n' % str(datetime.now()))
        self.f.write('Script path: %s\r\n' % str(self.path))
        self.f.write('#' * 80)
        self.f.write('\r\n')
        self.closefile()
        return self

    def make_log(self, **kwargs):
        self.init_log()
        self.f = open(os.path.join(self.path, 'LOG.txt'), 'a+')
        # self.f.write('Parameters\r\n')
        for k, v in zip(kwargs.keys(), kwargs.values()):
            self.f.write(f'{k} = {v}\r\n')
        self.f.write('\r\n')
        self.closefile()
        return self

    def openfile(self):
        self.f = open(os.path.join(self.path, 'LOG.txt'), 'a+')
        return self

    def closefile(self):
        self.f.close()

    def append_str(self, str_list: list):
        self.openfile()
        for s in str_list:
            self.f.write('%s\r\n' % s)
        self.closefile()
        return self

    def append_dict(self, kwargs):
        # TODO: fix
        self.openfile()
        for k, v in zip(kwargs.keys(), kwargs.values()):
            self.f.write(f'{k}={v} \t')
        self.f.write('\r\n')
        self.closefile()
        return self

    def append_cm(self, cm):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        self.openfile()
        self.f.write('\tCM (CM_norm): (tn fp // fn tp)')
        for i in range(cm.shape[0]):
            self.f.write('\r\n')
            for j in range(cm.shape[1]):
                self.f.write('\t %d (%.2f) \t' % (cm[i, j], cm_norm[i, j]))
        self.f.write('\r\n')
        self.f.write('\r\n')
        self.closefile()
        return self
