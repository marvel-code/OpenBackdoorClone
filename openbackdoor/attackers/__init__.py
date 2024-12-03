from .attacker import Attacker
from .ep_attacker import EPAttacker
from .sos_attacker import SOSAttacker
from .neuba_attacker import NeuBAAttacker
from .por_attacker import PORAttacker
from .lwp_attacker import LWPAttacker
from .lws_attacker import LWSAttacker
from .ripples_attacker import RIPPLESAttacker
from .orderbkd_attacker import OrderBkdAttacker

ATTACKERS = {
    "base": Attacker,
    "ep": EPAttacker,
    "sos": SOSAttacker,
    "neuba": NeuBAAttacker,
    "por": PORAttacker,
    "lwp": LWPAttacker,
    'lws': LWSAttacker,
    'ripples': RIPPLESAttacker,
    'orderbkd': OrderBkdAttacker
}




def load_attacker(config):
    return ATTACKERS[config["name"].lower()](**config)
