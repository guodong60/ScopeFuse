# coding=utf-8
"""
核心模块：把一个方法的(method_node, tokens)切分成
    token → statement(语句) → block(语句块) → function(整个方法)
四层体系里的前三层结构化信息。

设计规则(和之前Java/tree-sitter版本的思路一致，但用javalang的position信息实现，
不需要"花括号就近合并"那类特殊处理——因为我们只在[start_pos, end_pos)区间里取token，
落在区间之外的花括号/多余符号会自然被丢弃，不进入最终的token序列)：

  - 方法签名(修饰符/返回类型/方法名/参数/throws)单独成一条语句，属于block 0(block_type='sig')
  - 方法体顶层是block 1(block_type='top', nest_depth=0)
  - 普通语句(赋值/声明/return/方法调用等)：整体一条语句单元，不再拆分
  - 控制结构(if/for/while/do/try/switch)：
      · 头部(如"if (cond)"、"for(init;cond;update)"、"try")单独成一条语句单元，
        属于*外层*block(不嵌套)
      · 其body是一个*新的block*(nest_depth+1)，body内每条子语句各自独立成语句单元
      · else/catch/finally/case等"分支切换关键字"，各自单独成一条极简语句单元
        (如"else"、"catch (SQLException e)"、"case 1:")，属于外层block(不嵌套)，
        其分支体各自是一个新block(nest_depth+1)

  - block_type取值: 'sig','top','if','else','for','while','do','try','catch','finally','case','block'
    (最后的'block'对应独立花括号块，即不属于任何控制结构、单纯用来做变量作用域隔离的裸{...})
  - stmt_type取值: javalang节点类名转小写(如'statementexpression','localvariabledeclaration',
    'returnstatement')，控制结构头部/分支关键字用固定标记('if_head','else_head','for_head',
    'while_head','try_head','catch_head','finally_head','do_while_cond','switch_head','case_head','sig')

关键字定位采用"花括号深度感知"搜索(_find_keyword_from_idx及其三个变体)：扫描时遇到'{'
深度+1、'}'深度-1，只有深度回到0才允许匹配目标关键字(else/catch/finally/while/case)。
这是为了正确处理嵌套场景——比如外层if的then分支内部还嵌套了一个"有自己else"的if，
如果不做深度跟踪，会把内层的else误判成外层的(反之try/catch、do/while同理)。

已知限制(先做到能覆盖绝大多数情况，边界情况做保底而不是精细处理，避免一开始就过度设计):
  - switch的每个case目前不再细分表达式内部结构，case标签本身作为一条极简语句
  - try-with-resources(resources字段)、labeled statement 暂不特殊处理，会被当成普通语句整体处理
  - 极端的"无花括号嵌套else"边界情况(如`if(a) if(b) x(); else y(); else z();`全部不带花括号)
    理论上仍可能定位错误，但这种写法几乎不会出现在有代码规范的真实项目里，暂不特殊处理
  - 任何解析/切分过程中抛出的异常，都会被上层build_dataset.py捕获并跳过该样本(记录日志)，
    不应该在这里悄悄吞掉异常掩盖问题
"""
from java_parser import parse_method, JavaParseError

CONTROL_STMT_TYPES = ('IfStatement', 'ForStatement', 'WhileStatement',
                       'DoStatement', 'TryStatement', 'SwitchStatement')


def _pos_key(pos):
    return (pos.line, pos.column)


class StmtBlockBuilder:
    def __init__(self, tokens, include_modifiers=False):
        self.tokens = tokens
        # include_modifiers=True时，方法签名语句从"类壳后的第一个token"开始，
        # 这样public/private/static/@Override/<T>等修饰符、注解、泛型参数都会保留在签名里。
        # javalang的method_node.position指向返回类型(跳过了修饰符和注解)，默认行为会丢掉它们。
        self.include_modifiers = include_modifiers
        self._pos_list = [_pos_key(t.position) for t in tokens]
        self.stmt_records = []   # [{id, block_id, nest_depth, stmt_type, start, end}]
        self.block_records = []  # [{id, parent_block_id, nest_depth, block_type}]
        self.token2stmt = {}     # token_idx -> stmt_id

    # ---------- 基础工具 ----------
    def _idx_at_or_after(self, pos):
        """二分查找第一个position >= pos的token下标(tokens按源码顺序，position单调不减)"""
        key = _pos_key(pos)
        lo, hi = 0, len(self._pos_list)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._pos_list[mid] < key:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def _find_keyword_from_idx(self, values, start_idx, upper_bound):
        """核心的"花括号深度感知"关键字搜索：从start_idx(闭区间)开始扫描，
        遇到'{'深度+1、'}'深度-1，只有深度回到0时才允许匹配values里的关键字——
        这样能正确跳过途中任何嵌套{...}内部出现的同名关键字(比如内层if自己的else、
        内层try自己的catch)，不会把它们误认成外层的。upper_bound是位置上界(不含)。
        """
        before_key = _pos_key(upper_bound) if upper_bound is not None else None
        depth = 0
        for i in range(start_idx, len(self.tokens)):
            if before_key is not None and self._pos_list[i] >= before_key:
                return None
            val = self.tokens[i].value
            if val == '{':
                depth += 1
            elif val == '}':
                depth -= 1
            elif depth == 0 and val in values:
                return i
        return None

    def _find_keyword_after_scope(self, values, scope_start_pos, upper_bound):
        """找"给定scope整体结束之后"出现的关键字，例如then分支结束后的else、
        try块结束后的catch、do循环体结束后的while。scope_start_pos是这个scope
        自己的起始位置(如果scope带花括号就是它的'{'，否则是这条语句自己的起始token)。
        从这个位置(含)开始扫描：如果scope自己就是花括号包起来的，第一个token正好是
        它自己的'{'，会被正常计入深度(0→1)，之后途中所有嵌套结构自然被跳过，
        直到遇到scope自己的收尾'}'(深度归0)，紧随其后的关键字才会被匹配到。
        """
        start_idx = self._idx_at_or_after(scope_start_pos)
        return self._find_keyword_from_idx(values, start_idx, upper_bound)

    def _find_next_sibling_keyword(self, values, current_kw_idx, upper_bound):
        """"链式"查找：已知当前某个关键字(比如某个catch/case自己的关键字)的token下标，
        找下一个同级关键字(下一个catch/finally，或下一个case/default)。
        从current_kw_idx+1开始扫(跳过关键字自身，避免把自己当成"下一个")，
        深度处理同上，能正确跳过当前这个catch/case自己内部的嵌套结构。
        """
        return self._find_keyword_from_idx(values, current_kw_idx + 1, upper_bound)

    def _find_keyword_inside_scope(self, values, scope_brace_pos, upper_bound):
        """找"直接位于给定scope自己花括号内部、不跨越更深嵌套"的关键字，
        例如switch自己花括号内的顶层case/default标签(不会深入嵌套switch/block找)。
        scope_brace_pos是这个scope自己左花括号'{'的位置，从它*之后*(不含它本身)开始扫描。
        """
        brace_idx = self._idx_at_or_after(scope_brace_pos)
        return self._find_keyword_from_idx(values, brace_idx + 1, upper_bound)

    def _new_block(self, parent_block_id, nest_depth, block_type):
        bid = len(self.block_records)
        self.block_records.append(dict(id=bid, parent=parent_block_id,
                                        nest_depth=nest_depth, block_type=block_type))
        return bid

    def _new_stmt(self, block_id, nest_depth, stmt_type, start_pos, end_pos):
        start_idx = self._idx_at_or_after(start_pos)
        end_idx = self._idx_at_or_after(end_pos) if end_pos is not None else len(self.tokens)
        if start_idx >= end_idx:
            return None  # 空区间(比如空的else block之类)，跳过
        sid = len(self.stmt_records)
        self.stmt_records.append(dict(id=sid, block_id=block_id, nest_depth=nest_depth,
                                       stmt_type=stmt_type, start=start_idx, end=end_idx))
        for i in range(start_idx, end_idx):
            self.token2stmt[i] = sid
        return sid

    # ---------- 主入口 ----------
    def build(self, method_node):
        sig_block = self._new_block(None, 0, 'sig')
        body_list = method_node.body or []
        if body_list:
            body_start_pos = body_list[0].position
        else:
            body_start_pos = None  # 空方法体(极少见)，签名一路占到token末尾
        sig_start_pos = method_node.position
        if self.include_modifiers and len(self.tokens) > 3 and self.tokens[2].value == '{':
            # tokens[0:3]固定是类壳的'class' '_dummy_' '{'(见java_parser.WRAPPER_PREFIX)，
            # tokens[3]就是方法的第一个token(修饰符/注解/类型参数/返回类型之一)
            sig_start_pos = self.tokens[3].position
        self._new_stmt(sig_block, 0, 'sig', sig_start_pos, body_start_pos)

        top_block = self._new_block(None, 0, 'top')
        if body_list:
            self._walk_stmt_list(body_list, top_block, nest_depth=0, upper_bound=None)

    # ---------- 语句列表递归 ----------
    def _walk_stmt_list(self, stmts, block_id, nest_depth, upper_bound):
        """upper_bound: 这个列表里最后一条语句的结束边界(不含)，None表示直到token末尾"""
        n = len(stmts)
        for i, stmt in enumerate(stmts):
            next_bound = stmts[i + 1].position if i + 1 < n else upper_bound
            type_name = type(stmt).__name__
            if type_name in CONTROL_STMT_TYPES:
                self._walk_control_stmt(stmt, type_name, block_id, nest_depth, next_bound)
            elif type_name == 'BlockStatement':
                # 独立的花括号块(比如纯粹用来做变量作用域隔离的裸{...}，不属于任何
                # if/for/while等控制结构)，同样应该展开成一个新的嵌套block，
                # 而不是整体当成一条扁平语句(那样会丢失里面的语句级粒度)
                new_block = self._new_block(block_id, nest_depth + 1, 'block')
                self._walk_stmt_list(stmt.statements, new_block, nest_depth + 1, next_bound)
            else:
                self._new_stmt(block_id, nest_depth, type_name.lower(),
                               stmt.position, next_bound)

    # ---------- 控制结构分发 ----------
    def _walk_control_stmt(self, stmt, type_name, block_id, nest_depth, upper_bound):
        if type_name == 'IfStatement':
            self._walk_if(stmt, block_id, nest_depth, upper_bound)
        elif type_name in ('ForStatement', 'WhileStatement'):
            self._walk_for_while(stmt, type_name, block_id, nest_depth, upper_bound)
        elif type_name == 'DoStatement':
            self._walk_do(stmt, block_id, nest_depth, upper_bound)
        elif type_name == 'TryStatement':
            self._walk_try(stmt, block_id, nest_depth, upper_bound)
        elif type_name == 'SwitchStatement':
            self._walk_switch(stmt, block_id, nest_depth, upper_bound)

    def _body_as_list(self, body_node):
        """then_statement/body等字段可能是BlockStatement(有.statements)，
        也可能是单条非block语句(如`if(x) doA();`不带花括号)，统一成list处理"""
        if body_node is None:
            return []
        if type(body_node).__name__ == 'BlockStatement':
            return body_node.statements
        return [body_node]

    def _walk_if(self, stmt, block_id, nest_depth, upper_bound):
        then_list = self._body_as_list(stmt.then_statement)
        else_node = stmt.else_statement
        # then分支的边界：如果有else，就是else关键字的位置(用深度感知搜索，跳过then分支
        # 内部任何嵌套if自己的else)；否则是外层upper_bound
        else_kw_idx = None
        if else_node is not None:
            else_kw_idx = self._find_keyword_after_scope({'else'}, stmt.then_statement.position,
                                                          upper_bound)
        then_upper = self.tokens[else_kw_idx].position if else_kw_idx is not None else upper_bound

        self._new_stmt(block_id, nest_depth, 'if_head', stmt.position, stmt.then_statement.position)
        then_block = self._new_block(block_id, nest_depth + 1, 'if')
        self._walk_stmt_list(then_list, then_block, nest_depth + 1, then_upper)

        if else_node is not None:
            if else_kw_idx is not None:
                else_kw_pos = self.tokens[else_kw_idx].position
                self._new_stmt(block_id, nest_depth, 'else_head', else_kw_pos, else_node.position)
            # else if链: else_node本身还是IfStatement时，按同样逻辑递归(nest_depth+1，
            # 和普通嵌套if一致处理，不做"else if视为同级"的特殊优化，保持规则简单统一)
            if type(else_node).__name__ == 'IfStatement':
                self._walk_control_stmt(else_node, 'IfStatement', block_id, nest_depth + 1, upper_bound)
            else:
                else_list = self._body_as_list(else_node)
                else_block = self._new_block(block_id, nest_depth + 1, 'else')
                self._walk_stmt_list(else_list, else_block, nest_depth + 1, upper_bound)

    def _walk_for_while(self, stmt, type_name, block_id, nest_depth, upper_bound):
        kw = 'for_head' if type_name == 'ForStatement' else 'while_head'
        block_type = 'for' if type_name == 'ForStatement' else 'while'
        self._new_stmt(block_id, nest_depth, kw, stmt.position, stmt.body.position)
        body_list = self._body_as_list(stmt.body)
        new_block = self._new_block(block_id, nest_depth + 1, block_type)
        self._walk_stmt_list(body_list, new_block, nest_depth + 1, upper_bound)

    def _walk_do(self, stmt, block_id, nest_depth, upper_bound):
        # do{...} while(cond); 用深度感知搜索定位收尾的'while'，跳过body内部
        # 任何嵌套while/do自己的关键字(比如body里还有一个完整的while循环)
        while_kw_idx = self._find_keyword_after_scope({'while'}, stmt.body.position, upper_bound)
        body_upper = self.tokens[while_kw_idx].position if while_kw_idx is not None else upper_bound

        self._new_stmt(block_id, nest_depth, 'do_head', stmt.position, stmt.body.position)
        new_block = self._new_block(block_id, nest_depth + 1, 'do')
        body_list = self._body_as_list(stmt.body)
        self._walk_stmt_list(body_list, new_block, nest_depth + 1, body_upper)

        if while_kw_idx is not None:
            self._new_stmt(block_id, nest_depth, 'do_while_cond',
                           self.tokens[while_kw_idx].position, upper_bound)

    def _walk_try(self, stmt, block_id, nest_depth, upper_bound):
        catches = stmt.catches or []
        finally_list = stmt.finally_block  # 是list或None，不是node

        # try块本身的上界：往后找第一个catch或finally关键字，用深度感知搜索跳过
        # try块内部嵌套的另一个try-catch自己的catch/finally
        stop_idx = self._find_keyword_after_scope({'catch', 'finally'}, stmt.position, upper_bound)
        try_block_upper = self.tokens[stop_idx].position if stop_idx is not None else upper_bound

        first_stmt_pos = stmt.block[0].position if stmt.block else None
        self._new_stmt(block_id, nest_depth, 'try_head', stmt.position,
                       first_stmt_pos if first_stmt_pos else try_block_upper)
        try_block = self._new_block(block_id, nest_depth + 1, 'try')
        self._walk_stmt_list(stmt.block, try_block, nest_depth + 1, try_block_upper)

        # 逐个catch：cursor_idx链式指向"下一个catch/finally关键字"的token下标，
        # 每一步都用_find_next_sibling_keyword从当前关键字之后找，天然跳过当前
        # catch自己内部嵌套的try-catch
        cursor_idx = stop_idx
        for catch in catches:
            if cursor_idx is None:
                break
            catch_kw_pos = self.tokens[cursor_idx].position
            first_catch_stmt_pos = catch.block[0].position if catch.block else None

            next_stop_idx = self._find_next_sibling_keyword({'catch', 'finally'}, cursor_idx, upper_bound)
            catch_upper = self.tokens[next_stop_idx].position if next_stop_idx is not None else upper_bound

            self._new_stmt(block_id, nest_depth, 'catch_head', catch_kw_pos,
                           first_catch_stmt_pos if first_catch_stmt_pos else catch_upper)
            catch_block = self._new_block(block_id, nest_depth + 1, 'catch')
            self._walk_stmt_list(catch.block, catch_block, nest_depth + 1, catch_upper)
            cursor_idx = next_stop_idx

        if finally_list and cursor_idx is not None:
            fin_kw_pos = self.tokens[cursor_idx].position
            first_fin_stmt_pos = finally_list[0].position
            self._new_stmt(block_id, nest_depth, 'finally_head', fin_kw_pos, first_fin_stmt_pos)
            fin_block = self._new_block(block_id, nest_depth + 1, 'finally')
            self._walk_stmt_list(finally_list, fin_block, nest_depth + 1, upper_bound)

    def _find_first_brace(self, start_idx, upper_bound):
        """简单地找从start_idx开始第一个'{'token。注意不能复用_find_keyword_from_idx，
        因为那个函数把'{'当作深度计数符号处理(专门用来跳过，不会被当成"目标关键字"匹配)，
        这里要找的恰恰是'{'本身，属于另一种更简单的场景(不涉及嵌套跳过)。"""
        before_key = _pos_key(upper_bound) if upper_bound is not None else None
        for i in range(start_idx, len(self.tokens)):
            if before_key is not None and self._pos_list[i] >= before_key:
                return None
            if self.tokens[i].value == '{':
                return i
        return None

    def _walk_switch(self, stmt, block_id, nest_depth, upper_bound):
        cases = stmt.cases or []
        # 先定位switch自己的左花括号(此时还没进入任何嵌套，用简单扫描即可)
        brace_idx = self._find_first_brace(self._idx_at_or_after(stmt.position), upper_bound)
        if brace_idx is None:
            # 理论上switch必须带花括号，走到这里说明解析出了异常情况，保底整体当一条语句
            self._new_stmt(block_id, nest_depth, 'switch_head', stmt.position, upper_bound)
            return
        brace_pos = self.tokens[brace_idx].position

        # 顶层第一个case/default：必须"直接位于switch自己花括号内部"，用_inside_scope
        # 而不是_after_scope，避免跳过整个switch内容去外面找
        first_case_kw_idx = self._find_keyword_inside_scope({'case', 'default'}, brace_pos, upper_bound)
        header_end = self.tokens[first_case_kw_idx].position if first_case_kw_idx is not None else upper_bound
        self._new_stmt(block_id, nest_depth, 'switch_head', stmt.position, header_end)

        # 之后每个case到下一个case/default：链式查找，天然跳过当前case自己内部
        # 嵌套的switch/block
        cursor_idx = first_case_kw_idx
        for case in cases:
            if cursor_idx is None:
                break
            case_kw_pos = self.tokens[cursor_idx].position
            first_stmt_pos = case.statements[0].position if case.statements else None

            next_case_kw_idx = self._find_next_sibling_keyword({'case', 'default'}, cursor_idx, upper_bound)
            case_upper = self.tokens[next_case_kw_idx].position if next_case_kw_idx is not None else upper_bound

            self._new_stmt(block_id, nest_depth, 'case_head', case_kw_pos,
                           first_stmt_pos if first_stmt_pos else case_upper)
            case_block = self._new_block(block_id, nest_depth + 1, 'case')
            self._walk_stmt_list(case.statements, case_block, nest_depth + 1, case_upper)
            cursor_idx = next_case_kw_idx


def _is_brace(tok):
    return type(tok).__name__ == 'Separator' and tok.value in ('{', '}')


def _strip_structural_braces(tokens, builder):
    """去掉每条语句首尾的结构性花括号。

    背景：块结构已经由block体系显式表达，'{'/'}'不再提供信息，反而带来噪声：
      - '{'总是挂在头部语句末尾(如'try {'、签名末尾的'{')
      - '}'总是挂在block最后一条语句末尾(如'return x ; }')，方法最后一条语句甚至会带上
        类壳的收尾'}'，出现'throw ... ; } } }'这种三连花括号
      - do-while的'} while ( c ) ;'开头会带上do体的'}'
    只去掉语句首尾的花括号，语句内部的花括号(数组初始化、匿名类、lambda体)原样保留，
    因为这些语句总以';'结尾，不会被误删。每条语句至少保留1个token。
    """
    for rec in builder.stmt_records:
        s, e = rec['start'], rec['end']
        s2, e2 = s, e
        while e2 - s2 > 1 and _is_brace(tokens[e2 - 1]):
            e2 -= 1
        while e2 - s2 > 1 and _is_brace(tokens[s2]):
            s2 += 1
        for i in list(range(s, s2)) + list(range(e2, e)):
            if builder.token2stmt.get(i) == rec['id']:
                del builder.token2stmt[i]
        rec['start'], rec['end'] = s2, e2


def build_stmt_blocks(raw_code: str, include_modifiers=False, strip_braces=False):
    """
    对外主接口。输入原始code字符串，输出：
      code_tokens:  按源码顺序的token字符串列表(只包含被分配给某条语句的token，
                    壳token/游离花括号等已自然被排除)
      token_types:  与code_tokens等长，每个token对应的javalang类型名
                    (如'Identifier'/'String'/'DecimalInteger'/'Operator'/'Separator'等)，
                    code_tokenizer.py按这个类型决定是拆分成子词还是原样保留/替换成占位符
      token2stmt:   与code_tokens等长，每个token对应的stmt_id
      stmt_infos:   list，下标=stmt_id，每项 {block_id, nest_depth, stmt_type}
      block_infos:  list，下标=block_id，每项 {parent, nest_depth, block_type}
    """
    method_node, tokens = parse_method(raw_code)
    builder = StmtBlockBuilder(tokens, include_modifiers=include_modifiers)
    builder.build(method_node)
    if strip_braces:
        _strip_structural_braces(tokens, builder)

    code_tokens, token_types, token2stmt = [], [], []
    for idx in sorted(builder.token2stmt.keys()):
        code_tokens.append(tokens[idx].value)
        token_types.append(type(tokens[idx]).__name__)
        token2stmt.append(builder.token2stmt[idx])

    stmt_infos = [{'block_id': r['block_id'], 'nest_depth': r['nest_depth'], 'stmt_type': r['stmt_type']}
                  for r in builder.stmt_records]
    block_infos = [{'parent': r['parent'], 'nest_depth': r['nest_depth'], 'block_type': r['block_type']}
                   for r in builder.block_records]
    if len(code_tokens) == 0:
        raise ValueError('build_stmt_blocks返回了空token序列(可能是空方法体)')
    return code_tokens, token_types, token2stmt, stmt_infos, block_infos


def iter_build_stmt_blocks(data, code_field='code', show_progress=True, desc='stmt/block切分',
                           include_modifiers=False, strip_braces=False):
    """
    批量处理入口：所有需要"遍历数据集+调用build_stmt_blocks+抓异常+进度条"的脚本
    (validate_stmt_blocks.py、以后的build_dataset.py)都应该复用这一个函数，
    而不是各自重新写一遍循环——循环、进度条、异常处理只在这里维护一份。

    输入:
      data: list[dict]，每个元素至少包含code_field指定的字段(默认'code')
      show_progress: 是否显示tqdm进度条(没装tqdm时自动退化成无进度条遍历，不报错)
      desc: 进度条左侧显示的描述文字

    产出(生成器，逐条yield):
      (item, result, error)
      - 解析成功: result=(code_tokens, token2stmt, stmt_infos, block_infos)，error=None
      - 解析失败: result=None，error=对应的Exception实例(调用方可以用str(error)取信息)
    """
    iterator = data
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(data, desc=desc, unit='sample')
        except ImportError:
            iterator = data

    ok_count = 0
    for i, item in enumerate(iterator, start=1):
        code = item.get(code_field, '')
        try:
            result = build_stmt_blocks(code, include_modifiers=include_modifiers,
                                       strip_braces=strip_braces)
            ok_count += 1
            yield item, result, None
        except Exception as e:
            yield item, None, e
        if show_progress and hasattr(iterator, 'set_postfix'):
            iterator.set_postfix(ok='{:.1%}'.format(ok_count / i))