import logging
import sys
import matplotlib.pyplot as plt
import numpy as np


def plot_dot(m_indx, m_value, num=0, ):
    plt.plot(m_indx, m_value, 'ks')
    show_max = '[' + str(m_indx) + ',' + str("{:.4f}".format(m_value)) + ']'
    plt.annotate(show_max, xytext=(m_indx, m_value + num), xy=(m_indx, m_value))


def plot_base(train_c, valid_c, base_dir, mode, valid_c_3d=[], interval=0):
    train_x = range(len(train_c))
    train_y = train_c

    valid_x = range(len(valid_c))
    valid_y = valid_c
    plt.plot(train_x, train_y)
    plt.plot(valid_x, valid_y)
    if 'Loss' in mode:
        m_indx = np.argmin(valid_c)
    if 'Dice' in mode:
        m_indx = np.argmax(valid_c)
        m_indx_3d = np.argmax(valid_c_3d)
        valid_x_3d_init = list(range(len(valid_c_3d)))
        valid_x_3d = [x * interval for x in valid_x_3d_init]
        valid_y_3d = valid_c_3d
        plt.plot(valid_x_3d, valid_c_3d)
        if m_indx_3d < int(len(valid_c_3d) * 0.8):
            plot_dot(m_indx_3d * interval, valid_c_3d[m_indx_3d])
        last_indx_3d = valid_x_3d_init[-1]
        v_last_2d = valid_c[-1]
        v_last_3d = valid_c_3d[-1]
        abs_vLast = abs(v_last_3d - v_last_2d)
        if abs_vLast < 0.04:
            num = 0.04 - abs_vLast if v_last_3d > v_last_2d else -(0.06 - abs_vLast)
            plot_dot(last_indx_3d * interval, valid_y_3d[last_indx_3d], num)
        else:
            plot_dot(last_indx_3d * interval, valid_y_3d[last_indx_3d])
        plt.legend(['train', 'val', 'val_3d'], loc='upper left')
    else:
        plt.legend(['train', 'val'], loc='upper left')
    if m_indx < int(len(valid_c) * 0.8):
        plot_dot(m_indx, valid_c[m_indx])
    last_indx = valid_x[-1]
    plot_dot(last_indx, valid_c[last_indx])
    plt.ylabel(mode + ' value')
    plt.xlabel('epoch')
    plt.title("Model " + mode)
    plt.savefig('{}/{}-{:.4f}.jpg'.format(base_dir, mode, valid_c[m_indx]))
    plt.close()


def plot_dice_loss(loss_train_c, loss_valid_c, dice_train_c, dice_valid_c_2d, dice_valid_c_3d, val_3d_interval,
                   lr_curve, base_dir):
    plot_base(loss_train_c, loss_valid_c, base_dir, mode='Loss')
    plot_base(dice_train_c, dice_valid_c_2d, base_dir, 'Dice', dice_valid_c_3d, val_3d_interval)
    lr_x = range(len(lr_curve))
    lr_y = lr_curve
    plt.plot(lr_x, lr_y)
    plt.legend(['learning_rate'], loc='upper right')
    plt.ylabel('lr value')
    plt.xlabel('epoch')
    plt.title("Learning Rate")
    plt.savefig('{}/lr.jpg'.format(base_dir))
    plt.close()


def set_logging(log_path):
    logger = logging.getLogger()
    logger.setLevel('INFO')
    BASIC_FORMAT = '%(levelname)s: %(message)s'
    formatter = logging.Formatter(BASIC_FORMAT)
    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)
    fhlr = logging.FileHandler(log_path)
    fhlr.setFormatter(formatter)
    logger.addHandler(chlr)
    logger.addHandler(fhlr)


class Logger(object):
    def __init__(self, log_path="Default.log"):
        self.terminal = sys.stdout
        self.log = open(log_path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def read_list(list_path):
    list_data = []
    with open(list_path, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            list_data.append(line)
    return list_data



class AverageMeter(object):
    def __init__(self):
        self.value = 0
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.count_dict = {}
        self.value_dict = {}


        self.res_dict = {}

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count

    def add_value(self, tag_dict, n=1):
        for tag, value in tag_dict.items():
            if tag not in self.value_dict:
                self.value_dict[tag] = 0.0
                self.count_dict[tag] = 0
            self.value_dict[tag] += value * n
            self.count_dict[tag] += n

    def updata_avg(self):
        for tag in self.value_dict:
            if tag not in self.res_dict:
                self.res_dict[tag] = []
            avg_val = self.value_dict[tag] / self.count_dict[tag]
            self.res_dict[tag].append(avg_val)

        self.count_dict = {}
        self.value_dict = {}

    def reset(self):

        self.value = 0
        self.count = 0
        self.sum = 0
        self.avg = 0

        self.count_dict = {}
        self.value_dict = {}
        # self.res_dict = {}

