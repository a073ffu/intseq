TOKEN_ARG_COUNTS = {'0':0, '1':0, '2':0, 'x':0, 'plus':2, 'minus':2, 'multiply':2, 'division':2, 'mod':2, 'partial_sum':1,
                    'partial_sum_of_squares':1, 'self_convolution':1, 'linear_weighted_partial_sums':1, 'binomial':1, 'inverse_binomial_transform':1,
                    'product_of_two_consecutive_elements':1, 'cassini':1, 'first_stirling':1, 'second_stirling':1, 'first_differences':1,
                    'catalan':1, 'sum_of_divisors':1, 'moebius':1, 'hankel':1, 'boustrophedon':1}

EQUAL_TOKEN_WEIGHTS = {'0':1, '1':1, '2':1, 'x':1, 'plus':1, 'minus':1, 'multiply':1, 'division':1, 'mod':1, 'partial_sum':1,
                'partial_sum_of_squares':1, 'self_convolution':1, 'linear_weighted_partial_sums':1, 'binomial':1, 'inverse_binomial_transform':1,
                'product_of_two_consecutive_elements':1, 'cassini':1, 'first_stirling':0, 'second_stirling':0, 'first_differences':1,
                'catalan':1, 'sum_of_divisors':1, 'moebius':1, 'hankel':0, 'boustrophedon':1}

class Token_weights:
    def __init__(self, token_weights=EQUAL_TOKEN_WEIGHTS):
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