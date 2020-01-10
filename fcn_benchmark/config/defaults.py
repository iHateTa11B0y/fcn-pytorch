from yacs.config import CfgNode as CN

_C = CN()
_C.GPU = ()
_C.DIST_URL = "env://" #'tcp://224.66.41.62:23456'
_C.WORLD_SIZE = -1
_C.MULTIPROCESSING_DISTRIBUTED = True
_C.RANK = -1
_C.DIST_BACKEND = 'nccl'
_C.WORKERS = 8
_C.OUTPUT = 'weights'

_C.DATA = CN()
_C.DATA.TRAINSET = "/core1/data/home/niuwenhao/workspace/data/detection/door_all_new.json"
#_C.DATA.VAL_RATIO = 0.1
_C.DATA.TESTSET = "/core1/data/home/niuwenhao/data/tiny_bg.json"

_C.SOLVER = CN()
_C.SOLVER.LR = 0.02
_C.SOLVER.BATCH_SIZE = 4
_C.SOLVER.STEPS = (64000, 80000)
_C.SOLVER.EPOCH = 12
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 1e-4

_C.TEST_WEIGHT = '/home/niuwenhao/Repo/fcns/weights/model_epoch_9.pth'

