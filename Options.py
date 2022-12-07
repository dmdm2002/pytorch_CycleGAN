class param(object):
    def __init__(self):
        # Path
        self.ROOT = 'Z:/2nd_paper'
        self.DATASET_PATH = f'{self.ROOT}/dataset/ND/original'
        self.OUTPUT_CKP = f'{self.ROOT}/backup/CycleGAN/ckp'
        self.OUTPUT_SAMPLE = f'{self.ROOT}/backup/CycleGAN/sampling'
        self.OUTPUT_TEST = f'{self.ROOT}/backup/CycleGAN/test'
        self.OUTPUT_LOSS = f'{self.ROOT}/backup/CycleGAN/log'
        self.CKP_LOAD = False

        # Data
        self.DATA_STYPE = ['A', 'B']
        self.SIZE = 224
        self.POOL_SIZE = 50

        # Train or Test
        self.EPOCH = 300
        self.LR = 2e-4
        self.LAMDA_CYCLE = 10
        self.LAMDA_ID = 0.5
        self.BATCHSZ = 2

        # Handler
        # run_type 0 : train, 1 : test
        self.run_type = 0
