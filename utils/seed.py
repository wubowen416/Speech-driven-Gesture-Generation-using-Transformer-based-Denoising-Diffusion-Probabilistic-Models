def fix_seed(seed):
    """
    https://gist.github.com/Guitaricet/28fbb2a753b1bb888ef0b2731c03c031
    """
    import torch as th
    import numpy as np
    import random
    random.seed(seed)     # python random generator
    np.random.seed(seed)  # numpy random generator
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False