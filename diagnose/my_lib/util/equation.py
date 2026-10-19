import re
OPERATORS=[
    "/",
    "^",
    "+",
    "(",
    "-",
    "[",
    "]",
    ")",
    "*"
]

PREFIX_OPERATORS=[
    "/",
    "^",
    "+",
    "-",
    "*"
]

OP_ORDER_DIC={
    '+':0,
    '-':0,
    '*':1,
    '/':1,
    '^':2
}

N2A={'0':'a','1':'b','2':'c','3':'d','4':'e','5':'f','6':'g','7':'h','8':'i','9':'j'}
A2N= dict(zip(N2A.values(), N2A.keys()))


def is_number(string):
    '''
    判断当前字符串为数字，包括带括号的分数(3/4)
    :param string:
    :return:
    '''
    m = re.match(r"(\d+\.\d+\%)|(\d+\%)|(\d+\.\d+)|(\(\d+/\d+\))|(\d+/\d+)|(\d+)", string, re.S)
    if m is not None:
        if m.group() == string:
            return True
    return False

def is_int(string):
    '''
    判断当前字符串是否为整数
    :param string:
    :return:
    '''
    m = re.match(r"\d+", string, re.S)
    if m is not None:
        if m.group() == string:
            return True
    return False

def is_fraction(string):
    '''
    判断当前字符串为带括号的分数
    :param string:
    :return:
    '''
    m = re.match(r"(\(\d+/\d+\))|(\d+/\d+)", string, re.S)
    if m is not None:
        if m.group() == string:
            return True
    return False

def match_number(string):
    r = re.match(r"(\d+\.\d+\%)|(\d+\%)|(\d+\.\d+)|(\(\d+/\d+\))|(\d+/\d+)|(\d+)", string, re.S)
    if r is not None:
        return r.group()
    return None

def find_all_numbers(string):
    numbers = re.findall(r"\d+\.\d+\%|\d+\%|\d+\.\d+|\(\d+/\d+\)|\d+/\d+|\d+", string, re.S)
    # if r:   #避免冲突，把非整数替换
    #     string=re.sub(r"\d+\.\d+\%|\d+\%|\d+\.\d+|\(\d+/\d+\)|\d+/\d+",'<number>', string, re.S)
    # r.extend(re.findall(r'\d+',string,re.S))    #最后找整数
    unique_numbers=list(set(numbers))
    return sorted(unique_numbers,key=numbers.index) #去重后还要保持顺序

def numbers_to_identifiers(numbers):
    # assert is_number(string)
    # n2a={'0':'a','1':'b','2':'c','3':'d','4':'e','5':'f','6':'g','7':'h','8':'i','9':'j'}
    # identifiers = ['<n%d>' % j for j in range(len(numbers))]
    # identifiers=[[] for j in range(len(numbers))]
    identifiers=[]
    for i,number in enumerate(numbers):
        a=''.join([N2A[j] for j in str(i)])
        identifiers.append('<n%s>'%a)
    return identifiers

def identifier2sn(identifier):
    '''
    <nbc>-->12
    :param identifier:
    :return:
    '''
    sn_str=''
    for i in identifier[2:-1]:
        sn_str+=A2N[i]
    return int(sn_str)




def is_identifier(string):
    r=re.match(r'<n[a-j]+>', string, re.S)
    if r is not None and r.group()==string:
        return True
    return False

def is_identified_equation_token(string, prefix_operators=PREFIX_OPERATORS):
    '''
    判断该字符串是不是对数字标记处理后的属于公式的token，包括<n1>类、数字类和运算符
    :param string:
    :return:
    '''
    if is_identifier(string) or is_number(string) or string in prefix_operators:
        return True
    return False


#判断运算符的优先级
def opOrder(op1,op2,op_order_dic=OP_ORDER_DIC):
    '''
    比较运算符优先级
    :param op1:
    :param op2:
    :param op_order_dic: 优先级字典，如{'+':0,'-':0,'*':1,'/':1,'^':2}
    :return:
    '''
    # op_order_dic = {'*':4,'$':5,'/':4,'+':3,'-':3}
    if op1 in ['(','['] or op2 in ['(','[']:
        return False
    elif op2 in [')',']']:
        return True
    else:
        if op_order_dic[op1] < op_order_dic[op2]:
            return False
        else:
            return True

def _max_identifiers_numbers(ins):
    '''

    :param ins:
    :return:
    '''
    max_value=-1e10
    for inx in ins:
        if is_identifier(inx) and identifier2sn(inx)>max_value:
            max_value=identifier2sn(inx)
    return max_value if max_value!=-1e10 else 1e10

def _min_identifiers_numbers(ins):
    '''

    :param ins:
    :return:
    '''
    min_value=1e10
    for inx in ins:
        if is_identifier(inx) and identifier2sn(inx)<min_value:
            min_value=identifier2sn(inx)
    return min_value

def _sort_identifiers_numbers(ins):
    '''
    ['<n2>','<n1>','5']-->['<n1>',<n2>','5']
    :param ins:
    :return:
    '''
    identifiers=[]
    numbers=[]
    for inx in ins:
        if is_number(inx):
            numbers.append(inx)
        elif is_identifier(inx):
            identifiers.append(inx)
    assert len(identifiers+numbers)==len(ins)
    return sorted(identifiers,key=identifier2sn)+(numbers)

def infix2prefix(infix,operators=OPERATORS,op_order_dic=OP_ORDER_DIC):
    op_stack = []
    exp_stack = []
    for ch in infix:
        if not ch in operators:
            exp_stack.append(ch)
        elif ch in ['(','[']:
            op_stack.append(ch)
        elif ch == ')':
            while op_stack[-1] != '(':
                op = op_stack.pop()
                a = exp_stack.pop()
                b = exp_stack.pop()
                exp_stack.append(' '.join([op , b , a]))
            op_stack.pop()  # pop '('
        elif ch == ']':
            while op_stack[-1] != '[':
                op = op_stack.pop()
                a = exp_stack.pop()
                b = exp_stack.pop()
                exp_stack.append(' '.join([op , b , a]))
            op_stack.pop()  # pop '[
        else:
            while op_stack and op_stack[-1] not in ['(','['] and op_order_dic[ch] <= op_order_dic[op_stack[-1]]:
                op = op_stack.pop()
                a = exp_stack.pop()
                b = exp_stack.pop()
                exp_stack.append(' '.join([op , b , a]))
            op_stack.append(ch)

    # leftover
    while op_stack:
        op = op_stack.pop()
        a = exp_stack.pop()
        b = exp_stack.pop()
        exp_stack.append(' '.join([op , b , a]))
    # print(exp_stack[-1]
    return exp_stack[-1].split()

def infix2postfix(equation,operators=OPERATORS,op_order_dic=OP_ORDER_DIC):
    stack = []
    post_eq = []
    for elem in equation:
        if elem in ['(','[']:
            stack.append(elem)
        elif elem == ')':
            while 1:
                op = stack.pop()
                if op == '(':
                    break
                else:
                    post_eq.append(op)
        elif elem == ']':
            while 1:
                op = stack.pop()
                if op == '[':
                    break
                else:
                    post_eq.append(op)
        elif elem in operators:
            while 1:
                if stack == []:
                    break
                elif stack[-1] in ['(','[']:
                    break
                elif op_order_dic[elem] > op_order_dic[stack[-1]]:
                    break
                else:
                    op = stack.pop()
                    post_eq.append(op)
            stack.append(elem)
        else:
            #if elem == 'PI':
            #    post_eq.append('3.14')
            #else:
            #    post_eq.append(elem)
            post_eq.append(elem)
    while stack != []:
        post_eq.append(stack.pop())
    return post_eq

def _find_sub_prefix_end_id(start_id,prefix):
    '''
    对反向postfix，找到当前id对应的最小子式的终点
    :param start_id:
    :param rerversed_postfix:
    :return:
    '''
    if prefix[start_id] not in PREFIX_OPERATORS:
        return start_id
    else:
        op_num=1
        num_num=0
        for i,element in enumerate(prefix[start_id+1:]):
            if element in PREFIX_OPERATORS:
                op_num+=1
            else:
                num_num+=1
            if num_num>op_num:
                return start_id+i+1


def prefix_norm(prefix):
    '''
    rule1:在不增加前缀表达式长度的前提下去括号，如<na>+(<nb>+<nc>) -> <na>+<nb>+<nc>; <na>/(<nb>*<nc>) -> <na>/<nb>/<nc>；<na>*(<nb>+<nc>)不去括号
    rule2:尽可能缩短表达式，如<na>+<nb>-<nb> -> <na>; <na>+<nb>*<nc>/<nc> -> <na>+<nb>
    rule3:尽可能使数字按照它们在文本中出现的顺序出现，如<na>+<nbc>*<nb>+<nb> -> <na>+<nb>+<nb>*<nbc>
    :param equation:
    :return:
    '''
    # rule1
    i=0
    while i<len(prefix)-2:    #rule1
        if prefix[i]=='-':
            next_op_id=_find_sub_prefix_end_id(i+1,prefix)+1
            if prefix[next_op_id]=='-': #-<na>-*<nc><nd><ne> -> +-<na>*<nc><nd><ne>
                prefix[i+1:next_op_id+1]=prefix[i:next_op_id]
                prefix[i]='+'
                # continue
            elif prefix[next_op_id] == '+': #-<na>+*<nc><nd><ne> -> --<na>*<nc><nd><ne>
                prefix[i + 1:next_op_id + 1] = prefix[i:next_op_id]
                prefix[i] = '-'
                # continue
            else:
                i+=1
        elif prefix[i]=='+':
            next_op_id=_find_sub_prefix_end_id(i+1,prefix)+1
            if prefix[next_op_id]=='-': #+<na>-*<nc><nd><ne> -> -+<na>*<nc><nd><ne>
                prefix[i+1:next_op_id+1]=prefix[i:next_op_id]
                prefix[i]='-'
                # continue
            elif prefix[next_op_id] == '+': #+<na>+*<nc><nd><ne> -> ++<na>*<nc><nd><ne>
                prefix[i + 1:next_op_id + 1] = prefix[i:next_op_id]
                prefix[i] = '+'
                # continue
            else:
                i+=1
        elif prefix[i]=='/':
            next_op_id=_find_sub_prefix_end_id(i+1,prefix)+1
            if prefix[next_op_id]=='/': #/<na>/+<nc><nd><ne> -> */<na>+<nc><nd><ne>
                prefix[i+1:next_op_id+1]=prefix[i:next_op_id]
                prefix[i]='*'
                # continue
            elif prefix[next_op_id] == '*': #/<na>*+<nc><nd><ne> -> //<na>+<nc><nd><ne>
                prefix[i + 1:next_op_id + 1] = prefix[i:next_op_id]
                prefix[i] = '/'
                # continue
            else:
                i+=1
        elif prefix[i]=='*':
            next_op_id=_find_sub_prefix_end_id(i+1,prefix)+1
            if prefix[next_op_id]=='/':  #*<na>/+<nc><nd><ne> -> /*<na>+<nc><nd><ne>
                prefix[i+1:next_op_id+1]=prefix[i:next_op_id]
                prefix[i]='/'
                # continue
            elif prefix[next_op_id] == '*': # *<na>*+<nc><nd><ne> -> **<na>+<nc><nd><ne>
                prefix[i + 1:next_op_id + 1] = prefix[i:next_op_id]
                prefix[i] = '*'
                # continue
            else:
                i+=1
        else:
            i+=1

    # reversed rule3
    i=0
    while i<len(prefix)-2:    # reversed rule3
        sim_op_ids=[]
        if prefix[i] in ['+','-']:  #加减排序
            for j in range(i,len(prefix)):  #先收集+-队列
                if prefix[j] in ['+','-']:
                    sim_op_ids.append(j)
                else:
                    break   #最后一个符号肯定不是操作符，所以j都会在边界外
            if prefix[j-1]=='+':    #最后一个符号如果是+，比如 + <na> <nb> -> + <nb> <na>，，多一个冒泡
                seg_id1=_find_sub_prefix_end_id(j,prefix)+1
                seg_id2=_find_sub_prefix_end_id(j-1,prefix)+1
                if _max_identifiers_numbers(prefix[j:seg_id1])<=_min_identifiers_numbers(prefix[seg_id1:seg_id2]):
                    prefix[j:j+seg_id2-seg_id1],prefix[j+seg_id2-seg_id1:seg_id2]=prefix[seg_id1:seg_id2],prefix[j:seg_id1]
                # i+=1
            for m in range(1,len(sim_op_ids)):  #根据数字在文本中反向出现的顺序，对操作符和数字同时进行冒泡排序，操作符正向，数字反向
                for n in range(i,j-m):
                    seg_id1=_find_sub_prefix_end_id(n+2,prefix)+1
                    seg_id2=_find_sub_prefix_end_id(n+1,prefix)+1
                    seg_id3=_find_sub_prefix_end_id(n,prefix)+1
                    if _max_identifiers_numbers(prefix[seg_id1:seg_id2])<_min_identifiers_numbers(prefix[seg_id2:seg_id3])\
                        or (_max_identifiers_numbers(prefix[seg_id1:seg_id2])==_min_identifiers_numbers(prefix[seg_id2:seg_id3]) and
                            prefix[n]=='+' and prefix[n+1]=='-'):
                        prefix[seg_id1:seg_id1+seg_id3-seg_id2],prefix[seg_id1+seg_id3-seg_id2:seg_id3]=prefix[seg_id2:seg_id3],prefix[seg_id1:seg_id2]
                        prefix[n],prefix[n+1]=prefix[n+1],prefix[n]
                if prefix[j - 1] == '+':    #最后一个符号如果是+，比如 + <nb> <na> -> + <na> <nb>，多一个冒泡
                    seg_id1 = _find_sub_prefix_end_id(j, prefix) + 1
                    seg_id2 = _find_sub_prefix_end_id(j - 1, prefix) + 1
                    if _max_identifiers_numbers(prefix[j:seg_id1]) <= _min_identifiers_numbers(prefix[seg_id1:seg_id2]):
                        prefix[j:j + seg_id2 - seg_id1], prefix[j + seg_id2 - seg_id1:seg_id2] = prefix[seg_id1:seg_id2], prefix[j:seg_id1]
            i=j
        elif prefix[i] in ['*','/']:  # 乘除排序
            for j in range(i,len(prefix)):  #先收集*/队列
                if prefix[j] in ['*','/']:
                    sim_op_ids.append(j)
                else:
                    break   #最后一个符号肯定不是操作符，所以j都会在边界外
            if prefix[j-1]=='*':    #最后一个符号如果是*，比如 * <na> <nb> -> * <nb> <na>，，多一个冒泡
                seg_id1=_find_sub_prefix_end_id(j,prefix)+1
                seg_id2=_find_sub_prefix_end_id(j-1,prefix)+1
                if _max_identifiers_numbers(prefix[j:seg_id1])<=_min_identifiers_numbers(prefix[seg_id1:seg_id2]):
                    prefix[j:j+seg_id2-seg_id1],prefix[j+seg_id2-seg_id1:seg_id2]=prefix[seg_id1:seg_id2],prefix[j:seg_id1]
                # i+=1
            for m in range(1,len(sim_op_ids)):  #根据数字在文本中反向出现的顺序，对操作符和数字同时进行冒泡排序，操作符正向，数字反向
                for n in range(i,j-m):
                    seg_id1=_find_sub_prefix_end_id(n+2,prefix)+1
                    seg_id2=_find_sub_prefix_end_id(n+1,prefix)+1
                    seg_id3=_find_sub_prefix_end_id(n,prefix)+1
                    if _max_identifiers_numbers(prefix[seg_id1:seg_id2])<_min_identifiers_numbers(prefix[seg_id2:seg_id3])\
                        or (_max_identifiers_numbers(prefix[seg_id1:seg_id2])==_min_identifiers_numbers(prefix[seg_id2:seg_id3]) and
                            prefix[n]=='*' and prefix[n+1]=='/'):
                        prefix[seg_id1:seg_id1+seg_id3-seg_id2],prefix[seg_id1+seg_id3-seg_id2:seg_id3]=prefix[seg_id2:seg_id3],prefix[seg_id1:seg_id2]
                        prefix[n],prefix[n+1]=prefix[n+1],prefix[n]
                if prefix[j - 1] == '*':    #最后一个符号如果是*，比如 * <na> <nb> -> * <nb> <na>，多一个冒泡
                    seg_id1 = _find_sub_prefix_end_id(j, prefix) + 1
                    seg_id2 = _find_sub_prefix_end_id(j - 1, prefix) + 1
                    if _max_identifiers_numbers(prefix[j:seg_id1]) <= _min_identifiers_numbers(prefix[seg_id1:seg_id2]):
                        prefix[j:j + seg_id2 - seg_id1], prefix[j + seg_id2 - seg_id1:seg_id2] = prefix[seg_id1:seg_id2], prefix[j:seg_id1]
            i=j
        else:
            i+=1

    # rule2
    i = 0
    while i < len(prefix)-2:  #rule2
        if (prefix[i] == '+' and prefix[i + 1] == '-') or (prefix[i] == '-' and prefix[i + 1] == '+') \
                or (prefix[i] == '/' and prefix[i + 1] == '*') or (prefix[i] == '*' and prefix[i + 1] == '/'):
            seg_id1 = _find_sub_prefix_end_id(i + 2, prefix) + 1
            seg_id2 = _find_sub_prefix_end_id(i + 1, prefix) + 1
            seg_id3 = _find_sub_prefix_end_id(i, prefix) + 1
            if prefix[seg_id1:seg_id2] == prefix[seg_id2:seg_id3]:
                prefix = prefix[:i] + prefix[i + 2:seg_id1] + prefix[seg_id3:]
                if i-1>=0 and prefix[i-1] in ['+','-','*','-']:
                    i-=1
            else:
                i+=1
        else:
            i += 1

    # rule3
    i = 0
    while i < len(prefix) - 2:  # rule3
        sim_op_ids = []
        if prefix[i] in ['+', '-']:  # 加减排序
            for j in range(i, len(prefix)):  # 先收集+-队列
                if prefix[j] in ['+', '-']:
                    sim_op_ids.append(j)
                else:
                    break  # 最后一个符号肯定不是操作符，所以j都会在边界外
            if prefix[j - 1] == '+':  # 最后一个符号如果是+，比如 + <nb> <na> -> + <na> <nb>，，多一个冒泡
                seg_id1 = _find_sub_prefix_end_id(j, prefix) + 1
                seg_id2 = _find_sub_prefix_end_id(j - 1, prefix) + 1
                if _min_identifiers_numbers(prefix[j:seg_id1]) >= _max_identifiers_numbers(prefix[seg_id1:seg_id2]):
                    prefix[j:j + seg_id2 - seg_id1], prefix[j + seg_id2 - seg_id1:seg_id2] = prefix[
                                                                                             seg_id1:seg_id2], prefix[
                                                                                                               j:seg_id1]
                # i+=1
            for m in range(1, len(sim_op_ids)):  # 根据数字在文本中出现的顺序，对操作符和数字同时进行冒泡排序，操作符正向，数字反向
                for n in range(i, j - m):
                    seg_id1 = _find_sub_prefix_end_id(n + 2, prefix) + 1
                    seg_id2 = _find_sub_prefix_end_id(n + 1, prefix) + 1
                    seg_id3 = _find_sub_prefix_end_id(n, prefix) + 1
                    if _min_identifiers_numbers(prefix[seg_id1:seg_id2]) > _max_identifiers_numbers(
                            prefix[seg_id2:seg_id3]) \
                            or (_min_identifiers_numbers(prefix[seg_id1:seg_id2]) == _max_identifiers_numbers(
                        prefix[seg_id2:seg_id3]) and
                                prefix[n] == '-' and prefix[n + 1] == '+'):
                        prefix[seg_id1:seg_id1 + seg_id3 - seg_id2], prefix[
                                                                     seg_id1 + seg_id3 - seg_id2:seg_id3] = prefix[
                                                                                                            seg_id2:seg_id3], prefix[
                                                                                                                              seg_id1:seg_id2]
                        prefix[n], prefix[n + 1] = prefix[n + 1], prefix[n]
                if prefix[j - 1] == '+':  # 最后一个符号如果是+，比如 + <nb> <na> -> + <na> <nb>，多一个冒泡
                    seg_id1 = _find_sub_prefix_end_id(j, prefix) + 1
                    seg_id2 = _find_sub_prefix_end_id(j - 1, prefix) + 1
                    if _min_identifiers_numbers(prefix[j:seg_id1]) >= _max_identifiers_numbers(prefix[seg_id1:seg_id2]):
                        prefix[j:j + seg_id2 - seg_id1], prefix[j + seg_id2 - seg_id1:seg_id2] = prefix[
                                                                                                 seg_id1:seg_id2], prefix[
                                                                                                                   j:seg_id1]
            i = j
        elif prefix[i] in ['*', '/']:  # 乘除排序
            for j in range(i, len(prefix)):  # 先收集*/队列
                if prefix[j] in ['*', '/']:
                    sim_op_ids.append(j)
                else:
                    break  # 最后一个符号肯定不是操作符，所以j都会在边界外
            if prefix[j - 1] == '*':  # 最后一个符号如果是*，比如 * <nb> <na> -> * <na> <nb>，，多一个冒泡
                seg_id1 = _find_sub_prefix_end_id(j, prefix) + 1
                seg_id2 = _find_sub_prefix_end_id(j - 1, prefix) + 1
                if _min_identifiers_numbers(prefix[j:seg_id1]) >= _max_identifiers_numbers(prefix[seg_id1:seg_id2]):
                    prefix[j:j + seg_id2 - seg_id1], prefix[j + seg_id2 - seg_id1:seg_id2] = prefix[
                                                                                             seg_id1:seg_id2], prefix[
                                                                                                               j:seg_id1]
                # i+=1
            for m in range(1, len(sim_op_ids)):  # 根据数字在文本中出现的顺序，对操作符和数字同时进行冒泡排序，操作符正向，数字反向
                for n in range(i, j - m):
                    seg_id1 = _find_sub_prefix_end_id(n + 2, prefix) + 1
                    seg_id2 = _find_sub_prefix_end_id(n + 1, prefix) + 1
                    seg_id3 = _find_sub_prefix_end_id(n, prefix) + 1
                    if _min_identifiers_numbers(prefix[seg_id1:seg_id2]) > _max_identifiers_numbers(
                            prefix[seg_id2:seg_id3]) \
                            or (_min_identifiers_numbers(prefix[seg_id1:seg_id2]) == _max_identifiers_numbers(
                        prefix[seg_id2:seg_id3]) and
                                prefix[n] == '/' and prefix[n + 1] == '*'):
                        prefix[seg_id1:seg_id1 + seg_id3 - seg_id2], prefix[
                                                                     seg_id1 + seg_id3 - seg_id2:seg_id3] = prefix[
                                                                                                            seg_id2:seg_id3], prefix[
                                                                                                                              seg_id1:seg_id2]
                        prefix[n], prefix[n + 1] = prefix[n + 1], prefix[n]
                if prefix[j - 1] == '*':  # 最后一个符号如果是+，比如 + <nb> <na> -> + <na> <nb>，多一个冒泡
                    seg_id1 = _find_sub_prefix_end_id(j, prefix) + 1
                    seg_id2 = _find_sub_prefix_end_id(j - 1, prefix) + 1
                    if _min_identifiers_numbers(prefix[j:seg_id1]) >= _max_identifiers_numbers(prefix[seg_id1:seg_id2]):
                        prefix[j:j + seg_id2 - seg_id1], prefix[j + seg_id2 - seg_id1:seg_id2] = prefix[
                                                                                                 seg_id1:seg_id2], prefix[
                                                                                                                   j:seg_id1]
            i = j
        else:
            i += 1

    # rule2
    i = 0
    while i < len(prefix) - 2:  # rule2
        if (prefix[i] == '+' and prefix[i + 1] == '-') or (prefix[i] == '-' and prefix[i + 1] == '+') \
                or (prefix[i] == '/' and prefix[i + 1] == '*') or (prefix[i] == '*' and prefix[i + 1] == '/'):
            seg_id1 = _find_sub_prefix_end_id(i + 2, prefix) + 1
            seg_id2 = _find_sub_prefix_end_id(i + 1, prefix) + 1
            seg_id3 = _find_sub_prefix_end_id(i, prefix) + 1
            if prefix[seg_id1:seg_id2] == prefix[seg_id2:seg_id3]:
                prefix = prefix[:i] + prefix[i + 2:seg_id1] + prefix[seg_id3:]
                if i - 1 >= 0 and prefix[i - 1] in ['+', '-', '*', '-']:
                    i -= 1
            else:
                i += 1
        else:
            i += 1

    return prefix

if __name__=='__main__':
    infices = ["<na> / ( <nb> * <nc> )",
               "<nc> * ( <na> * <nb ) * <nb>",
               "<na> / ( <nc> * <nd> ) * <nc>",
               "<nd> / ( <nc> * ( <nb> + <na> ) ) * ( <nb> + <na> )",
               "<na> / [ <nc> * ( <nb> - <nd> ) ]",
               "<nd> - <na> / ( <na> + <nc> ) - <nd>",
               "<na> + <na> - <na> - <na> + <na>",
               "<na> + <nb> - <na>",
               "<nd> - <na> / ( <na> + <nc> ) - <nd>",
               "<nd> + <na> / ( <na> + <nd> ) - <nd>"]
    for i,infix in enumerate(infices):
        print('The infix {}:'.format(i),infix)
        prefix=infix2prefix(infix.split(), OPERATORS, OP_ORDER_DIC)
        print('The prefix {}: '.format(i),' '.join(prefix))
        norm_prefix=prefix_norm(prefix)
        print('The normalized prefix {}:'.format(i),' '.join(norm_prefix))

