from .rp_net import RP_Net
from .lgca_net_v3 import LGCANet_V3

model_factory = {
    'LGCANet_V3': LGCANet_V3,
    'RP_Net': RP_Net
}