from baselines.cpfair import CPFair
from baselines.fairsort import FairSort
from baselines.multifr import MultiFR
from baselines.ada2fair import Ada2Fair
from baselines.fair import FAIR

BASELINES = {
    'cpfair': CPFair,
    'fairsort': FairSort,
    'multifr': MultiFR,
    'ada2fair': Ada2Fair,
    'fair': FAIR,
}

CATEGORIES = {
    'cpfair': 'post-processing',
    'fairsort': 'post-processing',
    'multifr': 'in-processing',
    'ada2fair': 'in-processing',
    'fair': 'in-processing',
}

PROCESSING_TYPES = CATEGORIES
