# assert args.fusion_model in ['SAN', 'MLP', 'BAN', 'UD']

from .fusion_agcn import AGCN
from .fusion_ban import BAN
# from .fusion_bert import BERT
# from .fusion_bert_rnn import BERT
from .fusion_bert_rnn import BERT
from .fusion_lxmert import LXMERT
from .fusion_mlp import MLP
from .fusion_mlpq import MLPQ
from .fusion_san import SAN
from .fusion_updn import UD
