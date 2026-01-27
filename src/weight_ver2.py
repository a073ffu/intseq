TOKEN_ARG_COUNTS = {'0': 0,
                    '1': 0,
                    '2': 0,
                    'x': 0,
                    'n': 0,
                    'affine_-1_-1': 1,
                    'affine_-1_0': 1,
                    'affine_-1_1': 1,
                    'affine_1_-1': 1,
                    'affine_1_0': 1,
                    'affine_1_1': 1,
                    'affine_2_-1': 1,
                    'affine_2_0': 1,
                    'affine_2_1': 1,
                    'abs': 1,
                    'neg': 1,
                    'pow_2': 1,
                    'pow_3': 1,
                    'add': 2,
                    'sub': 2,
                    'mul': 2,
                    'sdiv': 2,
                    'smod': 2,
                    'shift_1': 1,
                    'shift_2': 1,
                    'shift_-1': 1,
                    'shift_-2': 1,
                    'dilate_2': 1,
                    'dilate_3': 1,
                    'diff': 1,
                    'diff2': 1,
                    'cumsum': 1,
                    'cumpowsum': 1,
                    'conv_diff': 1,
                    'conv_sum2': 1,
                    'conv_laplace': 1,
                    'ma_2': 1,
                    'ma_3': 1,
                    'dirichlet': 2,
                    'mu': 0,
                    'sigma_k0': 0,
                    'sigma_k1': 0,
                    'binomX': 1,
                    'binomInvX': 1,
                    'catalan': 1,
                    'boustrophedon': 1,
                    'alt': 0,
                    'poly_n': 0
                    }

TOKEN_RELATIVE_WEIGHTS = {'0': 0.06,
                    '1': 0.08,
                    '2': 0.04,
                    'x': 0.18,
                    'n': 0.12,
                    'affine_-1_-1': 0.01,
                    'affine_-1_0': 0.015,
                    'affine_-1_1': 0.01,
                    'affine_1_-1': 0.02,
                    'affine_1_0': 0.04,
                    'affine_1_1': 0.02,
                    'affine_2_-1': 0.01,
                    'affine_2_0': 0.015,
                    'affine_2_1': 0.01,
                    'abs': 0.008,
                    'neg': 0.03,
                    'pow_2': 0.02,
                    'pow_3': 0.01,
                    'add': 0.035,
                    'sub': 0.025,
                    'mul': 0.03,
                    'sdiv': 0.005,
                    'smod': 0.005,
                    'shift_1': 0.025,
                    'shift_2': 0.015,
                    'shift_-1': 0.025,
                    'shift_-2': 0.015,
                    'dilate_2': 0.025,
                    'dilate_3': 0.015,
                    'diff': 0.05,
                    'diff2': 0.02,
                    'cumsum': 0.05,
                    'cumpowsum': 0.02,
                    'conv_diff': 0.015,
                    'conv_sum2': 0.01,
                    'conv_laplace': 0.005,
                    'ma_2': 0.018,
                    'ma_3': 0.012,
                    'dirichlet': 0.008,
                    'mu': 0.01,
                    'sigma_k0': 0.02,
                    'sigma_k1': 0.02,
                    'binomX': 0,
                    'binomInvX': 0,
                    'catalan': 0,
                    'boustrophedon': 0,
                    'alt': 0.02,
                    'poly_n': 0.04}

class Token_weights:
    def __init__(self, token_weights=TOKEN_RELATIVE_WEIGHTS):
        self.token_weights = token_weights
        self.tokens = list(self.token_weights.keys())
        self.weights = list(self.token_weights.values())
        self.weight_sum = sum(self.weights)
        self.leaf_tokens = [k for k,v in TOKEN_ARG_COUNTS.items() if v == 0]
        self.leaf_weights = [v for k,v in self.token_weights.items() if k in self.get_leaf_tokens()]
        self.leaf_weight_sum = sum(self.get_leaf_weights())
        self.operator_tokens = [k for k, v in TOKEN_ARG_COUNTS.items() if v > 0]
        self.operator_weights = [self.token_weights[k] for k in self.operator_tokens]
        self.operator_weight_sum = sum(self.operator_weights)

    def set_Token_weights(self, token_weights:dict):
        self.token_weights = token_weights
        self.tokens = list(self.token_weights.keys())
        self.weights = list(self.token_weights.values())
        self.weight_sum = sum(self.weights)
        self.leaf_tokens = [k for k in self.token_weights if TOKEN_ARG_COUNTS[k] == 0]
        self.leaf_weights = [self.token_weights[k] for k in self.get_leaf_tokens()]
        self.leaf_weight_sum = sum(self.get_leaf_weights())
        self.operator_tokens = [k for k, v in TOKEN_ARG_COUNTS.items() if v > 0]
        self.operator_weights = [self.token_weights[k] for k in self.operator_tokens]
        self.operator_weight_sum = sum(self.operator_weights)

    
    def get_Token_weights(self):
        return self.token_weights
    
    def get_tokens(self):
        return self.tokens
    
    def get_weights(self):
        return self.weights
    
    def get_weight_sum(self):
        return self.weight_sum
    
    def get_leaf_tokens(self):
        return self.leaf_tokens
    
    def get_leaf_weights(self):
        return self.leaf_weights
    
    def get_leaf_weight_sum(self):
        return self.leaf_weight_sum
    
    def get_operator_tokens(self):
        return self.operator_tokens

    def get_operator_weights(self):
        return self.operator_weights
        
    def get_operator_weight_sum(self):
        return self.operator_weight_sum

#他のファイルで呼び出す
WEIGHTS = Token_weights()