from __future__ import annotations
from typing import List, Dict, Union
import math
import numpy as np
import sys
from sympy.ntheory import mobius, divisors
from sympy.functions.combinatorial.numbers import catalan, stirling
# 必要に応じてsympyのimportを追加

class SequenceError(Exception):
    def __init__(self, message="Insufficient elements for sequence generation."):
        super().__init__(message)

class Program:
    def __init__(self, **kwarg: Dict[str, Program]):
        self.sub_programs = kwarg

# ==========================================
# 1. 基本列生成トークン
# ==========================================

class Constant(Program):
    def __init__(self, k: int, numeric_sequence_length: int = 40):
        super().__init__()
        self.k = k
        self.numeric_sequence_length = numeric_sequence_length

    def calc(self, x: List[int]) -> List[int]:
        return [int(self.k)] * self.numeric_sequence_length

class Variable(Program):
    def __init__(self, name='x'):
        super().__init__()
        self.name = name

    def calc(self, x: List[int]) -> List[int]:
        if self.name == 'x':
            return [int(val) for val in x]
        return []

class Natural(Program):
    def __init__(self, numeric_sequence_length: int = 40):
        super().__init__()
        self.numeric_sequence_length = numeric_sequence_length

    def calc(self, x: List[int]) -> List[int]:
        # 入力xの長さ、もしくはデフォルト長に合わせて 0, 1, 2... を生成
        length = max(len(x), self.numeric_sequence_length)
        return list(range(length))

# ==========================================
# 2. 要素単位のアフィン変換
# ==========================================

class Affine_m1_m1(Program):
    def __init__(self, alpha: int = -1, beta: int = -1, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta

    def calc(self, x: List[int]) -> List[int]:
        seq_a = self.sub_programs['a'].calc(x)
        return [int(self.alpha * val + self.beta) for val in seq_a]

class Affine_m1_0(Program):
    def __init__(self, alpha: int = -1, beta: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta

    def calc(self, x: List[int]) -> List[int]:
        seq_a = self.sub_programs['a'].calc(x)
        return [int(self.alpha * val + self.beta) for val in seq_a]

class Affine_m1_p1(Program):
    def __init__(self, alpha: int = -1, beta: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta

    def calc(self, x: List[int]) -> List[int]:
        seq_a = self.sub_programs['a'].calc(x)
        return [int(self.alpha * val + self.beta) for val in seq_a]

class Affine_p1_m1(Program):
    def __init__(self, alpha: int = 1, beta: int = -1, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta

    def calc(self, x: List[int]) -> List[int]:
        seq_a = self.sub_programs['a'].calc(x)
        return [int(self.alpha * val + self.beta) for val in seq_a]

class Affine_p1_0(Program):
    def __init__(self, alpha: int = 1, beta: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta

    def calc(self, x: List[int]) -> List[int]:
        seq_a = self.sub_programs['a'].calc(x)
        return [int(self.alpha * val + self.beta) for val in seq_a]

class Affine_p1_p1(Program):
    def __init__(self, alpha: int = 1, beta: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta

    def calc(self, x: List[int]) -> List[int]:
        seq_a = self.sub_programs['a'].calc(x)
        return [int(self.alpha * val + self.beta) for val in seq_a]

class Affine_p2_m1(Program):
    def __init__(self, alpha: int = 2, beta: int = -1, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta

    def calc(self, x: List[int]) -> List[int]:
        seq_a = self.sub_programs['a'].calc(x)
        return [int(self.alpha * val + self.beta) for val in seq_a]

class Affine_p2_0(Program):
    def __init__(self, alpha: int = 2, beta: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta

    def calc(self, x: List[int]) -> List[int]:
        seq_a = self.sub_programs['a'].calc(x)
        return [int(self.alpha * val + self.beta) for val in seq_a]

class Affine_p2_p1(Program):
    def __init__(self, alpha: int = 2, beta: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta

    def calc(self, x: List[int]) -> List[int]:
        seq_a = self.sub_programs['a'].calc(x)
        return [int(self.alpha * val + self.beta) for val in seq_a]

class Abs(Program):
    def calc(self, x: List[int]) -> List[int]:
        seq_a = self.sub_programs['a'].calc(x)
        return [int(abs(val)) for val in seq_a]

class Neg(Program):
    def calc(self, x: List[int]) -> List[int]:
        seq_a = self.sub_programs['a'].calc(x)
        return [int(-val) for val in seq_a]

class Pow_2(Program):
    def __init__(self, p: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.p = p

    def calc(self, x: List[int]) -> List[int]:
        seq_a = self.sub_programs['a'].calc(x)
        # オーバーフロー防止のため実用的な範囲で計算、あるいはPythonの多倍長整数に任せる
        return [int(val ** self.p) for val in seq_a]


class Pow_3(Program):
    def __init__(self, p: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.p = p

    def calc(self, x: List[int]) -> List[int]:
        seq_a = self.sub_programs['a'].calc(x)
        # オーバーフロー防止のため実用的な範囲で計算、あるいはPythonの多倍長整数に任せる
        return [int(val ** self.p) for val in seq_a]

'''
# 重要度が低いため実装せず
# 不採用理由
# OEIS の数列定義として 極めて不自然
# 数学的意味が弱い
# 学習モデル向けの「都合の良い操作」になりやすい
class Clip(Program):
    def __init__(self, L: int = -100, U: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.L = L
        self.U = U

    def calc(self, x: List[int]) -> List[int]:
        seq_a = self.sub_programs['a'].calc(x)
        return [int(min(max(val, self.L), self.U)) for val in seq_a]
'''
        
# ==========================================
# 3. 二項演算（要素ごと）
# ==========================================

class Add(Program):
    def calc(self, x: List[int]) -> List[int]:
        seq_a = self.sub_programs['a'].calc(x)
        seq_b = self.sub_programs['b'].calc(x)
        length = min(len(seq_a), len(seq_b))
        if length == 0: raise SequenceError()
        return [int(seq_a[i] + seq_b[i]) for i in range(length)]

class Sub(Program):
    def calc(self, x: List[int]) -> List[int]:
        seq_a = self.sub_programs['a'].calc(x)
        seq_b = self.sub_programs['b'].calc(x)
        length = min(len(seq_a), len(seq_b))
        if length == 0: raise SequenceError()
        return [int(seq_a[i] - seq_b[i]) for i in range(length)]

class Mul(Program):
    def calc(self, x: List[int]) -> List[int]:
        seq_a = self.sub_programs['a'].calc(x)
        seq_b = self.sub_programs['b'].calc(x)
        length = min(len(seq_a), len(seq_b))
        if length == 0: raise SequenceError()
        return [int(seq_a[i] * seq_b[i]) for i in range(length)]

class SafeDiv(Program):
    def calc(self, x: List[int]) -> List[int]:
        seq_a = self.sub_programs['a'].calc(x)
        seq_b = self.sub_programs['b'].calc(x)
        length = min(len(seq_a), len(seq_b))
        if length == 0: raise SequenceError()
        
        res = []
        for i in range(length):
            denom = seq_b[i] if seq_b[i] != 0 else 1
            res.append(int(seq_a[i] // denom))
        return res

class SafeMod(Program):
    def calc(self, x: List[int]) -> List[int]:
        seq_a = self.sub_programs['a'].calc(x)
        seq_b = self.sub_programs['b'].calc(x)
        length = min(len(seq_a), len(seq_b))
        if length == 0: raise SequenceError()
        
        res = []
        for i in range(length):
            denom = seq_b[i] if seq_b[i] != 0 else 1
            res.append(int(seq_a[i] % denom))
        return res

# ==========================================
# 4. シフト・部分列変換
# ==========================================

class Shift_p1(Program):
    def __init__(self, k: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.k = k

    def calc(self, x: List[int]) -> List[int]:
        seq_a = self.sub_programs['a'].calc(x)
        length = len(seq_a)
        if self.k >= 0:
            # 右シフト (左側を0埋め)
            return [0] * min(self.k, length) + seq_a[:max(0, length - self.k)]
        else:
            # 左シフト: a_(n+k)
            # 先頭の要素が消え、短くなる
            # ★ 0埋め削除: 残った部分だけを返す
            shift = abs(self.k)
            if shift >= length:
                return []
            return seq_a[shift:]

class Shift_p2(Program):
    def __init__(self, k: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.k = k

    def calc(self, x: List[int]) -> List[int]:
        seq_a = self.sub_programs['a'].calc(x)
        length = len(seq_a)
        if self.k >= 0:
            # 右シフト (左側を0埋め)
            return [0] * min(self.k, length) + seq_a[:max(0, length - self.k)]
        else:
            # 左シフト: a_(n+k)
            # 先頭の要素が消え、短くなる
            # ★ 0埋め削除: 残った部分だけを返す
            shift = abs(self.k)
            if shift >= length:
                return []
            return seq_a[shift:]

class Shift_m1(Program):
    def __init__(self, k: int = -1, **kwargs):
        super().__init__(**kwargs)
        self.k = k

    def calc(self, x: List[int]) -> List[int]:
        seq_a = self.sub_programs['a'].calc(x)
        length = len(seq_a)
        if self.k >= 0:
            # 右シフト (左側を0埋め)
            return [0] * min(self.k, length) + seq_a[:max(0, length - self.k)]
        else:
            # 左シフト: a_(n+k)
            # 先頭の要素が消え、短くなる
            # ★ 0埋め削除: 残った部分だけを返す
            shift = abs(self.k)
            if shift >= length:
                return []
            return seq_a[shift:]

class Shift_m2(Program):
    def __init__(self, k: int = -2, **kwargs):
        super().__init__(**kwargs)
        self.k = k

    def calc(self, x: List[int]) -> List[int]:
        seq_a = self.sub_programs['a'].calc(x)
        length = len(seq_a)
        if self.k >= 0:
            # 右シフト (左側を0埋め)
            return [0] * min(self.k, length) + seq_a[:max(0, length - self.k)]
        else:
            # 左シフト: a_(n+k)
            # 先頭の要素が消え、短くなる
            # ★ 0埋め削除: 残った部分だけを返す
            shift = abs(self.k)
            if shift >= length:
                return []
            return seq_a[shift:]

class Dilate_2(Program):
    def __init__(self, k: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.k = k # k番目ごとの要素を残し、間は0

    def calc(self, x: List[int]) -> List[int]:
        seq_a = self.sub_programs['a'].calc(x)
        if self.k <= 0: return seq_a 
        res = []
        for n in range(len(seq_a)):
            if n % self.k == 0:
                idx = n // self.k
                if idx < len(seq_a):
                    res.append(seq_a[idx])
                else:
                    res.append(0)
            else:
                res.append(0)
        return res

class Dilate_3(Program):
    def __init__(self, k: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.k = k # k番目ごとの要素を残し、間は0

    def calc(self, x: List[int]) -> List[int]:
        seq_a = self.sub_programs['a'].calc(x)
        if self.k <= 0: return seq_a 
        res = []
        for n in range(len(seq_a)):
            if n % self.k == 0:
                idx = n // self.k
                if idx < len(seq_a):
                    res.append(seq_a[idx])
                else:
                    res.append(0)
            else:
                res.append(0)
        return res

# ==========================================
# 5. 累積・差分系
# ==========================================

class Diff(Program):
    def calc(self, x: List[int]) -> List[int]:
        seq_a = self.sub_programs['a'].calc(x)
        if len(seq_a) < 2: return []
        return [int(seq_a[i+1] - seq_a[i]) for i in range(len(seq_a)-1)]

class Diff2(Program):
    def calc(self, x: List[int]) -> List[int]:
        seq_a = self.sub_programs['a'].calc(x)
        if len(seq_a) < 3: return []
        # a_(n+2) - 2a_(n+1) + a_n
        return [int(seq_a[i+2] - 2 * seq_a[i+1] + seq_a[i]) for i in range(len(seq_a) - 2)]

class CumSum(Program):
    def calc(self, x: List[int]) -> List[int]:
        seq_a = self.sub_programs['a'].calc(x)
        res = []
        curr = 0
        for val in seq_a:
            curr += val
            res.append(int(curr))
        return res

class CumPowSum(Program):
    def __init__(self, p: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.p = p

    def calc(self, x: List[int]) -> List[int]:
        seq_a = self.sub_programs['a'].calc(x)
        res = []
        curr = 0
        for val in seq_a:
            curr += val ** self.p
            res.append(int(curr))
        return res

# ==========================================
# 6. 畳み込み（フィルタ系）
# ==========================================

class Conv_diff(Program):
    def __init__(self, kernel: List[int] = [1,-1], **kwargs):
        super().__init__(**kwargs)
        # デフォルトカーネル: [1, -1] (差分相当) など
        self.kernel = kernel if kernel is not None else [1, 1]

    def calc(self, x: List[int]) -> List[int]:
        seq_a = self.sub_programs['a'].calc(x)
        # numpyを使って畳み込み、サイズはseq_aに合わせる('same'ではないが定義に従う)
        # 定義: sum(a[n-i] * kernel[i]) -> numpy.convolveはfullを返す
        conv_res = np.convolve(seq_a, self.kernel, mode='full')
        return [int(val) for val in conv_res[:len(seq_a)]]

class Conv_sum2(Program):
    def __init__(self, kernel: List[int] = [1, 1], **kwargs):
        super().__init__(**kwargs)
        # デフォルトカーネル: [1, -1] (差分相当) など
        self.kernel = kernel if kernel is not None else [1, 1]

    def calc(self, x: List[int]) -> List[int]:
        seq_a = self.sub_programs['a'].calc(x)
        # numpyを使って畳み込み、サイズはseq_aに合わせる('same'ではないが定義に従う)
        # 定義: sum(a[n-i] * kernel[i]) -> numpy.convolveはfullを返す
        conv_res = np.convolve(seq_a, self.kernel, mode='full')
        return [int(val) for val in conv_res[:len(seq_a)]]

class Conv_laplace(Program):
    def __init__(self, kernel: List[int] = [1, -2, 1], **kwargs):
        super().__init__(**kwargs)
        # デフォルトカーネル: [1, -1] (差分相当) など
        self.kernel = kernel if kernel is not None else [1, 1]

    def calc(self, x: List[int]) -> List[int]:
        seq_a = self.sub_programs['a'].calc(x)
        # numpyを使って畳み込み、サイズはseq_aに合わせる('same'ではないが定義に従う)
        # 定義: sum(a[n-i] * kernel[i]) -> numpy.convolveはfullを返す
        conv_res = np.convolve(seq_a, self.kernel, mode='full')
        return [int(val) for val in conv_res[:len(seq_a)]]

class MovingAverage_w2(Program):
    def __init__(self, w: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.w = max(1, w)

    def calc(self, x: List[int]) -> List[int]:
        seq_a = self.sub_programs['a'].calc(x)
        res = []
        for n in range(len(seq_a)):
            start_idx = max(0, n - self.w + 1)
            window = seq_a[start_idx : n+1]
            if not window:
                res.append(0)
            else:
                res.append(int(sum(window) // len(window))) # 整数除算
        return res

class MovingAverage_w3(Program):
    def __init__(self, w: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.w = max(1, w)

    def calc(self, x: List[int]) -> List[int]:
        seq_a = self.sub_programs['a'].calc(x)
        res = []
        for n in range(len(seq_a)):
            start_idx = max(0, n - self.w + 1)
            window = seq_a[start_idx : n+1]
            if not window:
                res.append(0)
            else:
                res.append(int(sum(window) // len(window))) # 整数除算
        return res

# ==========================================
# 7. Dirichlet（約数和）系
# ==========================================

class DirichletConvolution(Program):
    def calc(self, x: List[int]) -> List[int]:
        seq_a = self.sub_programs['a'].calc(x)
        seq_b = self.sub_programs['b'].calc(x)
        length = min(len(seq_a), len(seq_b))
        res = [0] * length
        
        # Dirichlet convolution: (a * b)(n) = sum_{d|n} a(d) * b(n/d)
        # 配列インデックスnを自然数nとして扱う (index 0 は n=0に対応)
        # 通常n>=1で定義されるため、n=0は特別扱い（0とするか、a[0]*b[0]とするか等）
        # ここではn=0のとき0、n>=1について定義通り計算する
        
        if length > 0:
            res[0] = 0 # あるいは seq_a[0] * seq_b[0] など定義による
            
        for n in range(1, length):
            sum_val = 0
            # nの約数を求める
            divs = divisors(n)
            for d in divs:
                q = n // d
                if d < length and q < length:
                    sum_val += seq_a[d] * seq_b[q]
            res[n] = int(sum_val)
        return res

class MoebiusSeq(Program):
    def calc(self, x: List[int]) -> List[int]:
        # xの長さに応じたメビウス関数列を返す
        # 入力x自体は長さ参照にのみ使用（仕様によるが）
        length = len(x) if x else 20
        res = [0] * length
        for n in range(1, length):
            res[n] = int(mobius(n))
        return res

class Sigma_k0(Program):
    def __init__(self, k: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.k = k

    def calc(self, x: List[int]) -> List[int]:
        # 約数関数 sigma_k(n)
        length = len(x) if x else 20
        res = [0] * length
        for n in range(1, length):
            divs = divisors(n)
            sum_pow = sum([d ** self.k for d in divs])
            res[n] = int(sum_pow)
        return res

class Sigma_k1(Program):
    def __init__(self, k: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.k = k

    def calc(self, x: List[int]) -> List[int]:
        # 約数関数 sigma_k(n)
        length = len(x) if x else 20
        res = [0] * length
        for n in range(1, length):
            divs = divisors(n)
            sum_pow = sum([d ** self.k for d in divs])
            res[n] = int(sum_pow)
        return res

# ==========================================
# 8. 組合せ系
# ==========================================

class BinomialTransform(Program):
    def calc(self, x: List[int]) -> List[int]:
        seq_a = self.sub_programs['a'].calc(x)
        res = []
        for n in range(len(seq_a)):
            val = 0
            for k in range(n + 1):
                val += math.comb(n, k) * seq_a[k]
            res.append(int(val))
        return res

class InverseBinomialTransform(Program):
    def calc(self, x: List[int]) -> List[int]:
        seq_a = self.sub_programs['a'].calc(x)
        res = []
        for n in range(len(seq_a)):
            val = 0
            for k in range(n + 1):
                sign = (-1) ** (n - k)
                val += sign * math.comb(n, k) * seq_a[k]
            res.append(int(val))
        return res

class CatalanTransform(Program):
    def calc(self, x: List[int]) -> List[int]:
        seq_a = self.sub_programs['a'].calc(x)
        res = []
        # sum_{k=0..n} C_k * a_{n-k}
        c_vals = [int(catalan(i)) for i in range(len(seq_a))]
        
        for n in range(len(seq_a)):
            val = 0
            for k in range(n + 1):
                val += c_vals[k] * seq_a[n - k]
            res.append(int(val))
        return res

class BoustrophedonTransform(Program):
    def calc(self, x: List[int]) -> List[int]:
        seq_a = self.sub_programs['a'].calc(x)
        length = len(seq_a)
        if length == 0: return []

        # Entringer numbers E(n,k) を計算あるいは動的に生成
        # 定義: sum_{k=0..n} E(n,k) * a_k
        # E(n,k)はSeidel triangle (zigzag triangle) から得られる
        # ここでは変換を行うロジックを実装
        
        # Seidel Triangle Construction for Boustrophedon transform:
        # T[n,0] = a[n] (入力列を境界に置く変種もあるが、
        # ここでは公式 sum E(n,k)a_k に従うため、E(n,k)自体が必要)
        
        # Entringer数のテーブル作成 (必要サイズ分)
        # E(0,0)=1
        # E(n, 0) = 0 for n>0
        # E(n, k) = E(n, k-1) + E(n-1, n-k)
        
        E = np.zeros((length, length), dtype=int)
        E[0, 0] = 1
        for n in range(1, length):
            E[n, 0] = 0
            for k in range(1, n + 1):
                E[n, k] = E[n, k-1] + E[n-1, n-k]
                
        res = []
        for n in range(length):
            val = 0
            for k in range(n + 1):
                val += E[n, k] * seq_a[k]
            res.append(int(val))
        return res

# ==========================================
# 9. 補助列生成
# ==========================================

class Alternating(Program):
    def calc(self, x: List[int]) -> List[int]:
        # (-1)^n
        # 入力xは長さ参照用
        length = len(x) if x else 20
        return [1 if i % 2 == 0 else -1 for i in range(length)]

class PolyN(Program):
    def __init__(self, p: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.p = p

    def calc(self, x: List[int]) -> List[int]:
        # n^p
        length = len(x) if x else 20
        return [int(n ** self.p) for n in range(length)]

def check_if_constant_sequence(seq: List[Union[int, float]], n: int = None) -> bool:
    """
    与えられた数列が定数列であるかどうか、または先頭/末尾n個が定数列であるかをチェックします。
    nが指定されない場合、完全に定数列であるかチェックします。
    nが指定された場合、先頭n個または末尾n個が定数列であるかチェックします。
    """
    if not seq:
        return False

    if n is None: # nが指定されない場合、従来の完全な定数列チェック
        first_element = seq[0]
        for element in seq[1:]:
            if isinstance(element, float) or isinstance(first_element, float):
                if abs(element - first_element) > 1e-9:
                    return False
            else:
                if element != first_element:
                    return False
        return True
    else: # nが指定された場合、部分的な定数列チェック
        if n <= 0:
            raise ValueError("n must be a positive integer for partial constant sequence check.")
        if n > len(seq): # nが数列長を超える場合、全体チェックと同じ
            n = len(seq)

        # 先頭n個のチェック
        if n > 0: # nが0より大きい場合のみチェック
            first_element_head = seq[0]
            is_head_constant = True
            for i in range(1, n):
                if isinstance(seq[i], float) or isinstance(first_element_head, float):
                    if abs(seq[i] - first_element_head) > 1e-9:
                        is_head_constant = False
                        break
                else:
                    if seq[i] != first_element_head:
                        is_head_constant = False
                        break
            if is_head_constant:
                return True

        # 末尾n個のチェック
        if n > 0 and len(seq) >= n: # nが0より大きく、数列長がn以上の場合のみチェック
            first_element_tail = seq[len(seq) - n]
            is_tail_constant = True
            for i in range(len(seq) - n + 1, len(seq)):
                if isinstance(seq[i], float) or isinstance(first_element_tail, float):
                    if abs(seq[i] - first_element_tail) > 1e-9:
                        is_tail_constant = False
                        break
                else:
                    if seq[i] != first_element_tail:
                        is_tail_constant = False
                        break
            if is_tail_constant:
                return True
        
        return False # どちらも該当しない場合


def check_if_arithmetic_progression(seq: List[Union[int, float]]) -> bool:
    """
    与えられた数列が等差数列であるかどうかをチェックします。
    定数列も等差数列とみなされます。
    """
    if len(seq) < 2: # 1要素以下の数列は等差数列とみなさない（定義上はそうかもしれないが、実用上は除外しない）
        return False

    # 最初の公差を計算
    if isinstance(seq[1], float) or isinstance(seq[0], float):
        # 浮動小数点の場合
        diff = seq[1] - seq[0]
        # 定数列も等差数列に含まれるため、定数列チェック関数を呼び出す必要はない
        # ただし、浮動小数点誤差があるので注意
        for i in range(2, len(seq)):
            if abs((seq[i] - seq[i-1]) - diff) > 1e-9: # 小さい閾値を設ける
                return False
    else:
        # 整数値の場合
        diff = seq[1] - seq[0]
        for i in range(2, len(seq)):
            if (seq[i] - seq[i-1]) != diff:
                return False
    
    return True


# ==========================================
# Program Interpreter
# ==========================================

class ProgramInterpreter:
    # トークン文字列とクラスの対応
    # スカラパラメータが必要なクラスはデフォルト値でインスタンス化
    STR2CLASS = {
        'const': Constant, # default k=1
        'x': Variable,
        'n': Natural,
        'affine_-1_-1': Affine_m1_m1,
        'affine_-1_0': Affine_m1_0,
        'affine_-1_1': Affine_m1_p1,
        'affine_1_-1': Affine_p1_m1,
        'affine_1_0': Affine_p1_0,
        'affine_1_1': Affine_p1_p1,
        'affine_2_-1': Affine_p1_m1,
        'affine_2_0': Affine_p2_0,
        'affine_2_1': Affine_p2_p1,
        'abs': Abs,
        'neg': Neg,
        'pow_2': Pow_2,
        'pow_3': Pow_3,
        'add': Add,
        'sub': Sub,
        'mul': Mul,
        'sdiv': SafeDiv,
        'smod': SafeMod,
        'shift_1': Shift_p1,
        'shift_2': Shift_p2,
        'shift_-1': Shift_m1,
        'shift_-2': Shift_m2,
        'dilate_2': Dilate_2,
        'dilate_3': Dilate_3,
        'diff': Diff,
        'diff2': Diff2,
        'cumsum': CumSum,
        'cumpowsum': CumPowSum,
        'conv_diff': Conv_diff,
        'conv_sum2': Conv_sum2,
        'conv_laplace': Conv_laplace,
        'ma_2': MovingAverage_w2,
        'ma_3': MovingAverage_w3,
        'dirichlet': DirichletConvolution,
        'mu': MoebiusSeq,
        'sigma_k0': Sigma_k0,
        'sigma_k1': Sigma_k1,
        'binomX': BinomialTransform,
        'binomInvX': InverseBinomialTransform,
        'catalan': CatalanTransform,
        'boustrophedon': BoustrophedonTransform,
        'alt': Alternating,
        'poly_n': PolyN
    }

    STR2RPN_LIST = list(STR2CLASS.keys()) + ['0', '1', '2'] # 数字リテラルサポート用

    def __init__(self, rpn: List[str], numeric_sequence_length=20):
        self.rpn = rpn
        self.stack = []
        self.numeric_sequence_length = numeric_sequence_length
    
    @staticmethod
    def str2rpn(s: str) -> List[str]:
        # 簡易的な文字コードマッピング（元のコードの仕様）
        # 実際には文字列リストを直接渡す運用を想定
        return [] 

    def build(self):
        for s in self.rpn:
            # 定数リテラルの処理 (0, 1, 2)
            if s.isdigit():
                self.stack.append(Constant(int(s), self.numeric_sequence_length))
                continue

            if s not in self.STR2CLASS:
                raise ValueError(f"Unknown token: {s}")

            cls = self.STR2CLASS[s]
            
            # 引数の数（arity）に基づいてスタックからpopする
            # ※ ここではクラスごとにarityをハードコードするか、
            #    あるいはsub_programsの構造を動的に解決する必要がある
            #    簡略化のため、主要なクラスのarityで分岐する
            
            arity = 0
            if issubclass(cls, (Constant, Natural, MoebiusSeq, Sigma_k0, Sigma_k1, Alternating, PolyN, Variable)):
                arity = 0
            elif issubclass(cls, (Affine_m1_p1, Affine_m1_0, Affine_m1_m1, Affine_p1_m1, Affine_p1_0, Affine_p1_p1, Affine_p2_m1, Affine_p2_0, Affine_p2_p1, 
                                  Abs, Neg, Pow_2, Pow_3, Shift_p1, Shift_p2, Shift_m1, Shift_m2, 
                                  Dilate_2, Dilate_3, 
                                  Diff, Diff2, CumSum, CumPowSum, Conv_diff, Conv_sum2, Conv_laplace, MovingAverage_w2, MovingAverage_w3, 
                                  BinomialTransform, InverseBinomialTransform, 
                                  CatalanTransform, BoustrophedonTransform)):
                arity = 1
            elif issubclass(cls, (Add, Sub, Mul, SafeDiv, SafeMod, DirichletConvolution)):
                arity = 2
            
            # スタック操作
            kwargs = {}
            if arity == 1:
                kwargs['a'] = self.stack.pop()
            elif arity == 2:
                kwargs['b'] = self.stack.pop() # スタック順序注意: 後ろがb
                kwargs['a'] = self.stack.pop()
            
            # インスタンス化 (パラメータk, p等はデフォルト値を使用)
            # ※ 必要であればRPNからパラメータを読み取る拡張が必要
            instance = cls(**kwargs)
            
            # numeric_sequence_lengthなどの共通属性を注入
            if hasattr(instance, 'numeric_sequence_length'):
                instance.numeric_sequence_length = self.numeric_sequence_length
                
            self.stack.append(instance)
            
        return self.stack

# ==========================================
# Comprehensive Test Suite
# ==========================================

def run_tests():
    # テスト入力データ: 0~9の整数列
    x_input = list(range(10)) 
    print(f"--- Test Input x: {x_input} ---\n")
'''
    # テストケース: (Token Name, RPN List, Description)
    test_cases = [
        # --- 1. Basic ---
        ("Constant 5", ['5'], "Const(5)"),
        ("Variable x", ['x'], "Returns x"),
        ("Natural n", ['n'], "Returns 0, 1, 2..."),
        
        # --- 2. Affine (Unary) ---
        ("Affine", ['x', 'affine'], "1*x + 0 (Identity)"),
        ("Abs", ['x', 'neg', 'abs'], "abs(-x) -> x"),
        ("Neg", ['x', 'neg'], "-x"),
        ("Pow", ['x', 'pow'], "x^2 (default)"),
        ("Clip", ['x', 'clip'], "Clip -2 to 2"),

        # --- 3. Binary ---
        ("Add (x+2)", ['x', '2', 'add'], "x + 2"),
        ("Sub (x-x)", ['x', 'x', 'sub'], "x - x -> 0"),
        ("Mul (x*2)", ['x', '2', 'mul'], "x * 2"),
        ("SafeDiv (x/2)", ['x', '2', 'sdiv'], "x // 2"),
        ("SafeMod (x%3)", ['x', '3', 'smod'], "x % 3"),

        # --- 4. Shift/Seq ---
        ("Shift (Default k=1)", ['x', 'shift'], "Right shift"),
        ("Dilate (k=2)", ['x', 'dilate'], "Keep every 2nd elem"),
        ("Subseq", ['x', 'subseq'], "Start 0, step 2 -> 0, 2, 4..."),

        # --- 5. Cum/Diff ---
        ("Diff", ['x', 'diff'], "1, 1, 1... (Length reduced)"),
        ("Diff2", ['x', 'diff2'], "0, 0... (Length reduced)"),
        ("CumSum", ['x', 'cumsum'], "0, 1, 3, 6..."),
        ("CumPowSum", ['x', 'cumpowsum'], "Sum of squares"),

        # --- 6. Convolution ---
        ("Conv", ['x', 'conv'], "Convolve with [1, 1] -> x[n]+x[n-1]"),
        ("Moving Avg", ['x', 'ma'], "MA window=3"),

        # --- 7. Dirichlet ---
        # dirichlet(1, 1) は d(n) (約数の個数) になる
        ("Dirichlet (1*1)", ['1', '1', 'dirichlet'], "Divisor count function approx"),
        ("Moebius", ['mu'], "Moebius sequence"),
        ("Sigma", ['sigma'], "Sum of divisors"),

        # --- 8. Combinatorial ---
        ("Binomial", ['x', 'binomX'], "Binomial Transform"),
        ("InvBinomial", ['x', 'binomInvX'], "Inverse Binomial"),
        ("Catalan", ['x', 'catalan'], "Catalan Convolution"),
        ("Boustrophedon", ['1', 'boustrophedon'], "Tangent/Secant numbers approx"),

        # --- 9. Aux ---
        ("Alternating", ['alt'], "1, -1, 1, -1..."),
        ("PolyN", ['poly_n'], "n^2 (default)"),
    ]

    for name, rpn, desc in test_cases:
        try:
            interpreter = ProgramInterpreter(rpn)
            stack = interpreter.build()
            if not stack:
                print(f"[{name}] Error: Empty stack")
                continue
            
            program = stack[0]
            result = program.calc(x_input)
            
            # 長すぎる場合は一部省略
            res_str = str(result)
            if len(res_str) > 60:
                res_str = res_str[:55] + "...]"
            
            print(f"[{name:<15}] RPN:{str(rpn):<20} -> {res_str}")
            
        except Exception as e:
            print(f"[{name:<15}] FAILED: {e}")

if __name__ == "__main__":
    run_tests()
'''