#A=0 B=1 C=2 D=+ E=- F=* G=div H=mod I=cond J=loop K=x L=y M=compr N=loop2
import random
import math
import program


class ProgramGenerator(): 
    # トークンに応じた program_num の変化量を dict で定義
    VARIATION_OF_PROGRAM_NUM={'A':1, 'B':1, 'C':1, 'D':-1, 'E':-1, 'F':-1, 'G':-1, 'H':-1, 'I':-2, 'J':-2, 'K':1, 'L':1, 'M':-1, 'N':-4}
    
    def __init__(self):
        self.program_num=0 #リストをProgramStack()でオブジェクト化した時の要素数
        self.sequence=[]
        self.information_amount=0 #複雑度
    
    def append_letter(self, max_variation=1):
        letter, information_amount= self.generate_random_letter(1-self.program_num, max_variation)
        self.sequence.append(letter)
        self.update_program_num(letter)
        self.add_information_amount(information_amount)

    def generate_random_letter(self, min_variation, max_variation):
        letters=[]
        for k, v in self.VARIATION_OF_PROGRAM_NUM.item():
            if min_variation <= v <= max_variation:
                letters.append(k)
        
        return random.choice(letters), math.log(len(letters), 2)

    def add_information_amount(self, information_amount):
        self.information_amount+=information_amount

<<<<<<< HEAD
class Loop(Program):
    @staticmethod
    def loop_impl(f: Callable[[int, int], int], a: int, b: int) -> int:
        if a < 0:
            raise Exception("Invalid loop parameter: 'a' must be greater than or equal to 0.")
        while a > 0:
            b = f(b, a)
            a -= 1
        return b

    def build(self):
        if self.fn is not None:
            return self.fn
        
        ff = self.sub_programs['f'].build()
        fa = self.sub_programs['a'].build()
        fb = self.sub_programs['b'].build()
        
        self.fn = lambda x, y: self.loop_impl(ff, fa(x,y), fb(x,y))
        return self.fn

class Loop2(Program):
    @staticmethod
    def loop2_impl(f: Callable[[int, int], int],
                   g: Callable[[int, int], int],
                   a: int, b: int, c: int) -> int:
        if a < 0:
            raise Exception("Invalid loop2 parameter: 'a' must be greater than or equal to 0.")
        while a > 0:
            b, c = f(b, c), g(b,c)
            a -= 1
        return b

    def build(self) -> Callable[[int, int], int]:
        if self.fn is not None:
            return self.fn
        
        ff = self.sub_programs['f'].build()
        fg = self.sub_programs['g'].build()
        fa = self.sub_programs['a'].build()
        fb = self.sub_programs['b'].build()
        fc = self.sub_programs['c'].build()
        
        self.fn = lambda x, y: self.loop2_impl(ff, fg, fa(x,y), fb(x,y), fc(x,y))
        return self.fn

class Compr(Program):
    @staticmethod
    def compr_impl(f:Callable[[int, int], int], a:int, max_iter:int) -> int:
        if a == 0:
            m = 0
            while m < max_iter:
                if f(m, 0) <= 0:
                    return m
                m += 1
            raise Exception('compr_impl reached max_iter')
        elif a > 0:
            m = 0
            while m < max_iter:
                if m > Compr.compr_impl(f, a-1, max_iter) and f(m, 0) <= 0:
                    return m
                m += 1
            raise Exception('compr_impl reached max_iter')
        else:
            raise Exception('compr error')

    def build(self):
        if self.fn is not None:
            return self.fn
        
        ff = self.sub_programs['f'].build()
        fa = self.sub_programs['a'].build()

        max_iter = 10

        self.fn = lambda x, y: self.compr_impl(ff, fa(x,y), max_iter)
        return self.fn
    
class Plus(Program):
    def build(self):
        if self.fn is not None:
            return self.fn
        
        fa = self.sub_programs['a'].build()
        fb = self.sub_programs['b'].build()
        
        self.fn = lambda x, y: fa(x,y) + fb(x,y)
        return self.fn
        
class Minus(Program):
    def build(self):
        if self.fn is not None:
            return self.fn
        
        fa = self.sub_programs['a'].build()
        fb = self.sub_programs['b'].build()
        
        self.fn = lambda x, y: fa(x,y) - fb(x,y)
        return self.fn

class Multiply(Program):
    def build(self):
        if self.fn is not None:
            return self.fn
        
        fa = self.sub_programs['a'].build()
        fb = self.sub_programs['b'].build()
        
        self.fn = lambda x, y: fa(x,y) * fb(x,y)
        return self.fn

class Division(Program):
    def build(self):
        if self.fn is not None:
            return self.fn
        
        fa = self.sub_programs['a'].build()
        fb = self.sub_programs['b'].build()
        
        if lambda x, y: fb(x,y)==0:
            raise Exception("Division by zero")
        self.fn = lambda x, y: fa(x,y) // fb(x,y)
        return self.fn

class Mod(Program):
    def build(self):
        if self.fn is not None:
            return self.fn
        
        fa = self.sub_programs['a'].build()
        fb = self.sub_programs['b'].build()
        
        if lambda x, y: fb(x,y)==0:
            raise Exception("Mod by zero")
        self.fn = lambda x, y: fa(x,y) % fb(x,y)
        return self.fn

class ProgramStack:
    STR2RPN_LIST = ['0', '1', '2', 'plus', 'minus', 'multiply', 'div', 'mod', 'cond', 'loop', 'x', 'y', 'compr', 'loop2']

    def __init__(self, rpn):
        self.rpn = self.str2rpn(rpn)
        self.stack = []
    
    @staticmethod
    def str2rpn(str):
        return [ProgramStack.STR2RPN_LIST[ord(c)-ord('A')] for c in str]

    def build(self):
        for s in self.rpn:
            if isinstance(s, int) or s in ['0', '1', '2']:
                self.stack.append(Constant(int(s)))
            elif s in ['x', 'y']:
                self.stack.append(Variable(s))
            elif s == 'cond':
                c = self.stack.pop()
                b = self.stack.pop()
                a = self.stack.pop()
                self.stack.append(Cond(a=a, b=b, c=c))
            elif s == 'loop':
                b = self.stack.pop()
                a = self.stack.pop()
                f = self.stack.pop()
                self.stack.append(Loop(f=f, a=a, b=b))
            elif s == 'loop2':
                c = self.stack.pop()
                b = self.stack.pop()
                a = self.stack.pop()
                g = self.stack.pop()
                f = self.stack.pop()
                self.stack.append(Loop2(f=f, g=g, a=a, b=b, c=c))
            elif s == 'compr':
                a = self.stack.pop()
                f = self.stack.pop()
                self.stack.append(Compr(f=f, a=a))
            elif s == 'plus':
                b = self.stack.pop()
                a = self.stack.pop()
                self.stack.append(Plus(a=a, b=b))
            elif s == 'minus':
                b = self.stack.pop()
                a = self.stack.pop()
                self.stack.append(Minus(a=a, b=b))
            elif s == 'multiply':
                b = self.stack.pop()
                a = self.stack.pop()
                self.stack.append(Multiply(a=a, b=b))
            elif s == 'div':
                b = self.stack.pop()
                a = self.stack.pop()
                self.stack.append(Division(a=a, b=b))
            elif s == 'mod':
                b = self.stack.pop()
                a = self.stack.pop()
                self.stack.append(Mod(a=a, b=b))
        return self.stack
=======
    def update_program_num(self, r): #rの値に応じてprogram_numを増減
        self.program_num+=self.VARIATION_OF_PROGRAM_NUM[r]

    def program_num_info(self): # program_num を出力
        return self.program_num
    
    def sequence_info(self): # sequence を出力
        return self.sequence
    
    def information_amount_info(self): # information_amount を出力
        return self.information_amount

# 数列を生成
def generate(max_num_of_loops):
    sequence=ProgramGenerator()
    n=0
    loop_num=random.randint(1, max_num_of_loops)
    while(n < loop_num):
        sequence.append_letter()
        n+=1
        
    # sequneceのprogram_numを1にする
    while(sequence.program_num_info()>1):
        sequence.append_letter(-1)
    
    return sequence.sequence_info(), sequence.information_amount_info()

# 適切な数列を選別
def select_sequence(max_num_of_loops, max_iter=-1):
    while(True):
        try:
            sequence=[]
            information_amount=0
            sequence, information_amount=generate(max_num_of_loops)
            ps=program.ProgramStack(program.ProgramStack.str2rpn(sequence), max_iter)
            stack=ps.build()
            if check_if_y_is_bound(stack) and not check_if_constant_sequence(stack) and not check_if_trivial_arithmetic_progression(stack):
                return sequence, information_amount
            
        except Exception:
            continue

# yが束縛されているかどうかチェック
def check_if_y_is_bound(stack):
    return 'y' not in stack[0].find_free_variables()

# 定数数列かどうかチェック
def check_if_constant_sequence(stack):
    for i in range(1, 10):
        if (stack[0].calc(0, 0) != stack[0].calc(i, 0)):
            return False
    return True

# 初項0、公差1の等差数列(0,1,2,...)かどうかチェック
def check_if_trivial_arithmetic_progression(stack):
    for i in range(0, 10):
        if(stack[0].calc(i, 0) != i):
            return False
    return True
>>>>>>> develop
