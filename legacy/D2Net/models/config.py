import os
import json
import numpy as np

num_pool_per_axis = [5,5,5] #[5,5,5]  #ori: [5,5,5], or [4,4,4]
module = 'separable_adapter' # specific module type for universal model: series_adapter, parallel_adapter, separable_adapter
trainMode = 'universal'
task_idx = 0
residual = True #False
deep_supervision = True #False
deep_sup_type = 'add' # 'concat'  or 'add'

use_edge = False #True
use_dyrelu = False


