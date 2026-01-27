from __future__ import annotations
import math, random
from dataclasses import dataclass
from typing import Callable, List, Tuple, Dict, Set, Union
from collections import deque
import program_ver2 as program
import weight_ver2 as weight


# トークンごとの引数の数を定義
TOKEN_ARG_COUNTS: Dict[str, int] = {'0': 0,
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


# 数列変換操作を表す定数辞書
# 値は変換時の必要な追加長さまたは特別な文字列を示す
NUM_REDUCING_NUMERIC_SEQUENCE_LENGTH_AFTER_CALC: Dict[str, Union[int, str]] = {'0': 0,
                    '1': 0,
                    '2': 0,
                    'x': 0,
                    'n': 0,
                    'affine_-1_-1': 0,
                    'affine_-1_0': 0,
                    'affine_-1_1': 0,
                    'affine_1_-1': 0,
                    'affine_1_0': 0,
                    'affine_1_1': 0,
                    'affine_2_-1': 0,
                    'affine_2_0': 0,
                    'affine_2_1': 0,
                    'abs': 0,
                    'neg': 0,
                    'pow_2': 0,
                    'pow_3': 0,
                    'add': 0,
                    'sub': 0,
                    'mul': 0,
                    'sdiv': 0,
                    'smod': 0,
                    'shift_1': 0,
                    'shift_2': 0,
                    'shift_-1': 1,
                    'shift_-2': 2,
                    'dilate_2': 0,
                    'dilate_3': 0,
                    'diff': 1,
                    'diff2': 2,
                    'cumsum': 0,
                    'cumpowsum': 0,
                    'conv_diff': 0,
                    'conv_sum2': 0,
                    'conv_laplace': 0,
                    'ma_2': 0,
                    'ma_3': 0,
                    'dirichlet': 0,
                    'mu': 0,
                    'sigma_k0': 0,
                    'sigma_k1': 0,
                    'binomX': 0,
                    'binomInvX': 0,
                    'catalan': 0,
                    'boustrophedon': 0,
                    'alt': 0,
                    'poly_n': 0
                    }

class Node:
    def __init__(self, token, child_nodes:Node = None):
        self.token = token
        self.child_nodes = child_nodes if child_nodes is not None else []
    
    def set_token(self, token: str):
        assert token in TOKEN_ARG_COUNTS.keys()
        self.token = token

    def append_child_node(self, node: Node):
        self.child_nodes.append(node)
    
    def get_child_node(self, i: int):
        return self.child_nodes[i]

    def generate_program(self):
        child_tokens = []
        for child_node in self.child_nodes:
            child_tokens += child_node.generate_program()
        return child_tokens + [self.token]
    
    def check_is_x_bounded(self):
        is_x_bounded = False

        if any(child.check_is_x_bounded() for child in self.child_nodes):
            is_x_bounded = True
        
        elif self.token == 'x':
            is_x_bounded = True        
        
        return is_x_bounded


class ProgramGenerator:
    
    def __init__(self, max_depth:int):
        self.max_depth : int = max_depth
        self.information_amount : int = 0
        self.root_node : Node = None
    
    def select_random_tokens(self, num:int, is_leaf=False, must_have_args=False):
        '''
        引数の制限が一致しているトークンの中から
        weight.WEIGHTS.get_Token_weights()に応じた確率でnum個のトークンを返す
        - is_leaf: 葉トークン（引数なし）のみを選択
        - must_have_args: 引数を取るトークン（演算子）のみを選択
        選択されたトークンリストと、その選択に使用されたトークンセットの合計重みを返す。
        '''
        
        if is_leaf: # 葉ノード（深さ上限到達時）
            tokens = random.choices(weight.WEIGHTS.get_leaf_tokens(), weights=weight.WEIGHTS.get_leaf_weights(), k=num)
            return tokens, weight.WEIGHTS.get_leaf_weight_sum() # 葉ノードの合計重み
        elif must_have_args: # 強制パスで、深さ上限未到達時、引数を取るトークンのみを選択
            # weight.pyから演算子のトークンと重みを取得
            valid_operator_tokens = weight.WEIGHTS.get_operator_tokens()
            valid_operator_weights = weight.WEIGHTS.get_operator_weights()
            operator_weight_sum = weight.WEIGHTS.get_operator_weight_sum()

            if not valid_operator_tokens:
                raise ValueError("No operator tokens available with positive weights. Cannot build tree to specified depth.")

            tokens = random.choices(valid_operator_tokens, weights=valid_operator_weights, k=num)
            return tokens, operator_weight_sum # 演算子の合計重み
        else: # 通常の選択（引数なしトークンも含む）
            tokens = random.choices(weight.WEIGHTS.get_tokens(), weights=weight.WEIGHTS.get_weights(), k=num)
            return tokens, weight.WEIGHTS.get_weight_sum() # 全トークンの合計重み
    
    def build_tree(self, node=None, depth=0, force_deep_path=True): # 新しい引数 force_deep_path を追加
        # 1. ルートノードの処理
        if node is None:
            if self.max_depth < 1:
                raise ValueError("max_depth must be at least 1 for a valid program tree.")
            
            # ルートノードは常に引数を取るトークンから選択 (深さ1でツリーが終わるのを防ぐ)
            # max_depthが1の場合、ルートが演算子なら、その子ノードは強制的に葉になる
            token, current_weight_sum = self.select_random_tokens(1, must_have_args=True)
            self.root_node = Node(token[0])
            self.add_information_amount(math.log(current_weight_sum / weight.WEIGHTS.get_Token_weights()[token[0]], 2))
            
            # ここで force_deep_path を True にして最初の呼び出しを行う
            self.build_tree(self.root_node, depth + 1, force_deep_path=True) 
            
        # 2. 子ノードの処理
        else:
            # 現在の深さが目標の最大深さに達しているかチェック
            if depth == self.max_depth:
                # 最大深さに達したら、必ず葉ノード（引数なしのトークン）のみを選択
                arg_num = TOKEN_ARG_COUNTS[node.token]
                if arg_num == 0: # 既に葉ノードになっている場合（論理的にはありえない）
                    return
                
                tokens, current_weight_sum = self.select_random_tokens(arg_num, is_leaf=True) # 葉ノードのみを選択
                for token in tokens:
                    self.add_information_amount(math.log(current_weight_sum / weight.WEIGHTS.get_Token_weights()[token], 2))
                    child_node = Node(token)
                    node.append_child_node(child_node)
                return # この深さでツリー構築は終了
            
            # 最大深さに達していない場合
            arg_num = TOKEN_ARG_COUNTS[node.token]
            if arg_num == 0:
                # このケースは、force_deep_path が True のパスでは発生しないはず
                # force_deep_path が False のパスで、たまたま葉ノードが選ばれた場合はOK
                return # この枝の構築はここで終了

            # 子ノードを生成する前に、どのトークンタイプを選ぶか決定
            # force_deep_path が True の場合、少なくとも1つの子ノードは引数を取るトークンでなければならない
            # それ以外の子ノードは、葉ノードを含む任意のトークンを選択できる

            # 子ノードのリストを準備
            child_tokens_to_build = []
            
            # 強制的に深くなるパス (force_deep_path が True の場合のみ)
            if force_deep_path:
                # 少なくとも1つの子ノードは引数を取るトークンでなければならない
                # このパスに選ばれるトークンは演算子のみ
                operator_token_for_deep_path, operator_weight_sum = self.select_random_tokens(1, must_have_args=True)
                child_tokens_to_build.append({'token': operator_token_for_deep_path[0], 'force_deep': True, 'weight_sum_for_info': operator_weight_sum})
                
                # 残りの子ノードはランダムな深さでOK
                remaining_args = arg_num - 1
                for _ in range(remaining_args):
                    random_token, total_weight_sum = self.select_random_tokens(1) # 通常の選択なので全トークンの合計重み # 引数なしも含む
                    child_tokens_to_build.append({'token': random_token[0], 'force_deep': False, 'weight_sum_for_info': total_weight_sum})
            else:
                # force_deep_path が False の場合、全ての子ノードはランダムな深さでOK
                for _ in range(arg_num):
                    random_token, total_weight_sum = self.select_random_tokens(1) # 通常の選択なので全トークンの合計重み
                    child_tokens_to_build.append({'token': random_token[0], 'force_deep': False, 'weight_sum_for_info': total_weight_sum})

            # シャッフルしてランダム性を確保
            random.shuffle(child_tokens_to_build)

            # 各子ノードを再帰的に構築
            for item in child_tokens_to_build:
                token = item['token']
                force_deep_for_child = item['force_deep']
                current_weight_sum = item['weight_sum_for_info'] # 適切な合計重みを取得

                # 情報量の加算
                # 情報量計算に取得した current_weight_sum を使用
                self.add_information_amount(math.log(current_weight_sum / weight.WEIGHTS.get_Token_weights()[token], 2))
                
                child_node = Node(token)
                node.append_child_node(child_node)
                
                # トークンが引数を取る場合のみ再帰呼び出し
                if TOKEN_ARG_COUNTS[token] > 0:
                    self.build_tree(child_node, depth + 1, force_deep_path=force_deep_for_child)
                # 引数を取らない場合（葉ノード）は、ここで再帰は終了し、ツリーの枝がここで終わる
    
    
    # トークン名の変換テーブル（アルファベット→トークン名）
    TOKEN_MAP = [
        '0', '1', '2', 'x', 'plus', 'minus', 'multiply', 
        'division', 'mod', 'partial_sum', 'partial_sum_of_squares',
        'self_convolution', 'linear_weighted_partial_sums', 'binomial',
        'inverse_binomial_transform', 'product_of_two_consecutive_elements',
        'cassini', 'first_stirling', 'second_stirling', 'first_differences',
        'catalan', 'sum_of_divisors', 'moebius', 'hankel', 'boustrophedon'
    ]

    @classmethod
    def _convert_to_tokens(cls, rpn_string: str) -> List[str]:
        """アルファベットの文字列をトークン名のリストに変換

        Args:
            rpn_string: アルファベットで表現されたトークン列

        Returns:
            トークン名のリスト
        """
        return [cls.TOKEN_MAP[ord(c) - ord('A')] for c in rpn_string]

    def convert_token_sequence_to_tree(self, token_sequence:list):
        """トークン列からツリー構造を構築

        Returns:
            構築されたツリーの根ノード

        Raises:
            ValueError: トークン列が不正な場合
        """
        token_tree_stack = []

        for token in token_sequence:
            token_tree_stack = self._process_token(token_tree_stack, token)

        if len(token_tree_stack) != 1:
            raise ValueError("Invalid token sequence: Stack should contain exactly one node after processing")

        self.root_node = token_tree_stack[0]

    def _process_token(self, token_tree_stack:list, token: str) -> None:
        """個々のトークンを処理してスタックを更新

        Args:
            token: 処理対象のトークン
        """
        arg_count = TOKEN_ARG_COUNTS[token]

        if arg_count == 0:
            # 定数・変数の場合
            token_tree_stack.append(Node(token=token))
        else:
            # 演算子の場合、必要な数の引数を取り出してノードを作成
            if len(token_tree_stack) < arg_count:
                raise ValueError(f"Not enough arguments for token {token}")
                
            child_nodes = [token_tree_stack.pop() for _ in range(arg_count)]
            child_nodes.reverse()  # 引数の順序を元に戻す
            token_tree_stack.append(Node(token=token, child_nodes=child_nodes))
        
        return token_tree_stack
    

    def add_information_amount(self, information_amount): # 引数の値だけinformation_amountに加算
        self.information_amount += information_amount
    
    def get_token_sequence(self):
        return self.root_node.generate_program()
    
    def get_information_amount(self):
        return self.information_amount
    
    def check_is_x_bounded(self):
        return self.root_node.check_is_x_bounded()

def calculate_original_sequence_length(
    tree_node: Node, 
    necessary_numeric_sequence_length: int
) -> int:
    """トークンツリーから、目標の数列長を得るために必要な入力数列の長さを計算する

    Args:
        tree_node (Node): トークン木の各ノード
        necessary_numeric_sequence_length (int): 必要とする数列の長さ

    Returns:
        int: necessary_numeric_sequence_lengthの長さの数列を生成するのに入力数列の長さ

    Note:
        - トークンの種類によって以下の2つの計算方法で処理:
          1. 固定長減少: 特定のトークンで数列長が一定数減少
          2. Hankel変換: 数列長が約1/2になる
        - 2つの入力数列を使用する場合は、短い方の長さに合わせられる
    """
    # トークンに応じた長さの調整
    if isinstance(NUM_REDUCING_NUMERIC_SEQUENCE_LENGTH_AFTER_CALC[tree_node.token], int):
        # 固定長減少の場合
        adjusted_length = necessary_numeric_sequence_length + NUM_REDUCING_NUMERIC_SEQUENCE_LENGTH_AFTER_CALC[tree_node.token]
    elif NUM_REDUCING_NUMERIC_SEQUENCE_LENGTH_AFTER_CALC[tree_node.token] == 'Hankel':
        # Hankel変換の場合
        adjusted_length = necessary_numeric_sequence_length * 2
    else:
        assert False

    # 子ノードがない場合は計算終了
    if TOKEN_ARG_COUNTS[tree_node.token] == 0:
        return adjusted_length

    # 子ノードがある場合、再帰的に計算
    max_child_length = 0
    for child_node in tree_node.child_nodes:
        child_length = calculate_original_sequence_length(child_node, adjusted_length)
        max_child_length = max(max_child_length, child_length)
    
    return max_child_length