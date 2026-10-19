# # coding=utf-8
# """
# 分词模块：负责把stmt_block_builder吐出来的"原始token"(如 M_PriceList_ID、getContextAsDate、
# "SELECT ..."、42)进一步拆分成词表友好的子词序列，以及把摘要文本(text字段)分词。

# 设计原则：先用最简单直接的规则跑通，不追求一步到位(标识符按驼峰/下划线拆，字面量按类型
# 归一化成占位符)。等主干流程跑通、看到实际效果之后，再考虑要不要做更精细的处理(比如
# 拆分字符串字面量里的英文单词、区分短字面量和长字面量等)。

# 处理策略(按javalang token类型区分，类型名来自stmt_block_builder.build_stmt_blocks的token_types)：
#   - Identifier / BasicType / Modifier / Keyword: 按驼峰+下划线拆分成子词，全部转小写
#       例: M_PriceList_ID -> [m, price, list, id]
#           getContextAsDate -> [get, context, as, date]
#   - String(javalang把字符串字面量和字符字面量都归到这个类型，用首字符区分):
#       · 字符字面量(单引号开头，如'x') -> 占位符 <char>
#       · 字符串字面量(双引号开头) -> 占位符 <str>
#     这一步先不去拆字符串内容里的单词(比如不把"SELECT ... FROM ..."拆成select/from这些词)，
#     避免一开始就引入大量与代码结构无关的噪声词，等后面看效果决定要不要精细化
#   - 数字类字面量(DecimalInteger/DecimalFloatingPoint/HexInteger/OctalInteger等) -> 占位符 <num>
#   - Boolean(true/false) -> 保留原值(小写)，本身就是很小的封闭词表，没必要占位符化
#   - Null -> 保留'null'
#   - Operator / Separator / Annotation等符号类 -> 原样保留，不做任何拆分/转换
# """
# import re

# # 驼峰/连续大写(如HTTPRequest)拆分正则：
# #   [A-Z]+(?=[A-Z][a-z])  -> 处理"HTTPRequest"里的"HTTP"(后面跟着"Request"这种大写+小写的边界)
# #   [A-Z]?[a-z]+          -> 处理普通单词/驼峰单词，如"Price"、"list"
# #   [A-Z]+                -> 处理纯大写缩写，如"ID"、"URL"
# #   \d+                   -> 数字片段单独切出来，如"utf8" -> "utf","8"
# _CAMEL_SPLIT_RE = re.compile(r'[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+')

# _NUMERIC_TOKEN_TYPES = {
#     'DecimalInteger', 'DecimalFloatingPoint', 'HexInteger', 'HexFloatingPoint',
#     'OctalInteger', 'BinaryInteger', 'Integer', 'FloatingPoint',
# }

# STR_PLACEHOLDER = '<str>'
# CHAR_PLACEHOLDER = '<char>'
# NUM_PLACEHOLDER = '<num>'


# def split_identifier(token: str):
#     """把一个标识符拆成小写子词列表。先按下划线/连字符分组，组内再做驼峰拆分。
#     拆分失败(比如纯符号、空字符串)时兜底返回[token.lower()]，保证不丢token。
#     """
#     if not token:
#         return []
#     parts = re.split(r'[_\-]+', token)
#     subtokens = []
#     for p in parts:
#         if not p:
#             continue
#         subtokens.extend(_CAMEL_SPLIT_RE.findall(p))
#     if not subtokens:
#         return [token.lower()]
#     return [s.lower() for s in subtokens]


# def tokenize_code_token(value: str, token_type: str):
#     """对单个原始token按类型分发到不同处理策略，返回子词列表(可能是1个或多个)"""
#     if token_type in ('Identifier', 'BasicType', 'Modifier', 'Keyword'):
#         return split_identifier(value)
#     if token_type == 'String':
#         if len(value) >= 2 and value[0] == "'":
#             return [CHAR_PLACEHOLDER]
#         return [STR_PLACEHOLDER]
#     if token_type in _NUMERIC_TOKEN_TYPES:
#         return [NUM_PLACEHOLDER]
#     if token_type == 'Boolean':
#         return [value.lower()]
#     if token_type == 'Null':
#         return ['null']
#     # Operator/Separator/Annotation等符号类，原样保留(不转小写，避免"->"这类符号被误改)
#     return [value]


# def subtokenize_code(code_tokens, token_types, token2stmt):
#     """
#     对stmt_block_builder的输出做整体子词化，同时把token2stmt对齐扩展
#     (一个原始token拆成多个子词时，这几个子词共享同一个stmt_id)。

#     输入: code_tokens, token_types, token2stmt (三者等长，来自build_stmt_blocks)
#     输出: (subtokens, sub2stmt) 两个等长列表
#     """
#     subtokens, sub2stmt = [], []
#     for value, ttype, stmt_id in zip(code_tokens, token_types, token2stmt):
#         pieces = tokenize_code_token(value, ttype)
#         subtokens.extend(pieces)
#         sub2stmt.extend([stmt_id] * len(pieces))
#     return subtokens, sub2stmt


# # ---------- 摘要文本(text字段)分词 ----------
# _TEXT_WORD_RE = re.compile(r'[A-Za-z0-9_]+')


# def tokenize_text(text: str):
#     """摘要文本分词：转小写，只保留字母数字下划线组成的词，标点/多余空格直接丢弃。
#     例: " Runs a command on the command line synchronously."
#         -> [runs, a, command, on, the, command, line, synchronously]
#     """
#     if not text:
#         return []
#     return _TEXT_WORD_RE.findall(text.lower())


# if __name__ == '__main__':
#     # 自测：标识符拆分
#     tests = ['M_PriceList_ID', 'getContextAsDate', 'HTTPRequestHandler',
#               'utf8Encoder', 'i', 'p_WindowNo', 'retValue', 'DB']
#     for t in tests:
#         print(t, '->', split_identifier(t))

#     print()
#     # 自测：结合stmt_block_builder跑一条真实样本
#     from stmt_block_builder import build_stmt_blocks
#     code = ('private int findPLV(int M_PriceList_ID){\n'
#             '  Timestamp priceDate=null;\n'
#             '  String dateStr=Env.getContext(Env.getCtx(),p_WindowNo,"DateOrdered");\n'
#             '  int retValue=0;\n'
#             '  return retValue;\n'
#             '}\n')
#     code_tokens, token_types, token2stmt, stmt_infos, block_infos = build_stmt_blocks(code)
#     subtokens, sub2stmt = subtokenize_code(code_tokens, token_types, token2stmt)
#     print('原始token数:', len(code_tokens), '-> 子词数:', len(subtokens))
#     print(' '.join(subtokens))

#     print()
#     print('文本分词:', tokenize_text(' Find Price List Version and update context'))

# coding=utf-8
"""
分词模块：负责把stmt_block_builder吐出来的"原始token"(如 M_PriceList_ID、getContextAsDate、
"SELECT ..."、42)进一步拆分成词表友好的子词序列，以及把摘要文本(text字段)分词。

设计原则：先用最简单直接的规则跑通，不追求一步到位(标识符按驼峰/下划线拆，字面量按类型
归一化成占位符)。等主干流程跑通、看到实际效果之后，再考虑要不要做更精细的处理(比如
拆分字符串字面量里的英文单词、区分短字面量和长字面量等)。

处理策略(按javalang token类型区分，类型名来自stmt_block_builder.build_stmt_blocks的token_types)：
  - Identifier / BasicType / Modifier / Keyword: 按驼峰+下划线拆分成子词，全部转小写
      例: M_PriceList_ID -> [m, price, list, id]
          getContextAsDate -> [get, context, as, date]
  - String(javalang把字符串字面量和字符字面量都归到这个类型，用首字符区分):
      · 字符字面量(单引号开头，如'x') -> 占位符 <char>
      · 字符串字面量(双引号开头) -> 占位符 <str>
    这一步先不去拆字符串内容里的单词(比如不把"SELECT ... FROM ..."拆成select/from这些词)，
    避免一开始就引入大量与代码结构无关的噪声词，等后面看效果决定要不要精细化
  - 数字类字面量(DecimalInteger/DecimalFloatingPoint/HexInteger/OctalInteger等) -> 占位符 <num>
  - Boolean(true/false) -> 保留原值(小写)，本身就是很小的封闭词表，没必要占位符化
  - Null -> 保留'null'
  - Operator / Separator / Annotation等符号类 -> 原样保留，不做任何拆分/转换
"""
import re

# 驼峰/连续大写(如HTTPRequest)拆分正则：
#   [A-Z]+(?=[A-Z][a-z])  -> 处理"HTTPRequest"里的"HTTP"(后面跟着"Request"这种大写+小写的边界)
#   [A-Z]?[a-z]+          -> 处理普通单词/驼峰单词，如"Price"、"list"
#   [A-Z]+                -> 处理纯大写缩写，如"ID"、"URL"
#   \d+                   -> 数字片段单独切出来，如"utf8" -> "utf","8"
_CAMEL_SPLIT_RE = re.compile(r'[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+')

_NUMERIC_TOKEN_TYPES = {
    'DecimalInteger', 'DecimalFloatingPoint', 'HexInteger', 'HexFloatingPoint',
    'OctalInteger', 'BinaryInteger', 'Integer', 'FloatingPoint',
}

STR_PLACEHOLDER = '<str>'
CHAR_PLACEHOLDER = '<char>'
NUM_PLACEHOLDER = '<num>'


def split_identifier(token: str):
    """把一个标识符拆成小写子词列表。先按下划线/连字符分组，组内再做驼峰拆分。
    拆分失败(比如纯符号、空字符串)时兜底返回[token.lower()]，保证不丢token。
    """
    if not token:
        return []
    parts = re.split(r'[_\-]+', token)
    subtokens = []
    for p in parts:
        if not p:
            continue
        subtokens.extend(_CAMEL_SPLIT_RE.findall(p))
    if not subtokens:
        return [token.lower()]
    return [s.lower() for s in subtokens]


def tokenize_code_token(value: str, token_type: str):
    """对单个原始token按类型分发到不同处理策略，返回子词列表(可能是1个或多个)"""
    if token_type in ('Identifier', 'BasicType', 'Modifier', 'Keyword'):
        return split_identifier(value)
    if token_type == 'String':
        if len(value) >= 2 and value[0] == "'":
            return [CHAR_PLACEHOLDER]
        return [STR_PLACEHOLDER]
    if token_type in _NUMERIC_TOKEN_TYPES:
        return [NUM_PLACEHOLDER]
    if token_type == 'Boolean':
        return [value.lower()]
    if token_type == 'Null':
        return ['null']
    # Operator/Separator/Annotation等符号类，原样保留(不转小写，避免"->"这类符号被误改)
    return [value]


def subtokenize_code(code_tokens, token_types, token2stmt):
    """
    对stmt_block_builder的输出做整体子词化，同时把token2stmt对齐扩展
    (一个原始token拆成多个子词时，这几个子词共享同一个stmt_id)。

    输入: code_tokens, token_types, token2stmt (三者等长，来自build_stmt_blocks)
    输出: (subtokens, sub2stmt) 两个等长列表
    """
    subtokens, sub2stmt = [], []
    for value, ttype, stmt_id in zip(code_tokens, token_types, token2stmt):
        pieces = tokenize_code_token(value, ttype)
        subtokens.extend(pieces)
        sub2stmt.extend([stmt_id] * len(pieces))
    return subtokens, sub2stmt


# ---------- 摘要文本(text字段)分词 ----------
_TEXT_WORD_RE = re.compile(r'[A-Za-z0-9_]+')


# 文本模式：
#   'raw'          = 原始摘要，按上面的规则分词(小写、丢标点)。默认。
#   'pretokenized' = 摘要已经由HetCoS官方预处理做完分词/词形还原(例如 token_data 里的text:
#                    "run a command on the command line synchronously .." )，直接按空格切开，
#                    保留标点token和句末的".."结束标记，这样目标摘要和论文口径完全一致。
# 用模块级开关而不是给十几个函数加参数；由各个main()在开头调用set_text_mode。
_TEXT_MODE = 'raw'


def set_text_mode(mode: str):
    global _TEXT_MODE
    assert mode in ('raw', 'pretokenized'), mode
    _TEXT_MODE = mode


def get_text_mode():
    return _TEXT_MODE


def tokenize_text(text: str):
    """摘要文本分词。
    raw模式：转小写，只保留字母数字下划线组成的词，标点/多余空格直接丢弃。
      例: " Runs a command on the command line synchronously."
          -> [runs, a, command, on, the, command, line, synchronously]
    pretokenized模式：直接按空白切分，不做任何其他处理。
    """
    if not text:
        return []
    if _TEXT_MODE == 'pretokenized':
        return text.split()
    return _TEXT_WORD_RE.findall(text.lower())


if __name__ == '__main__':
    # 自测：标识符拆分
    tests = ['M_PriceList_ID', 'getContextAsDate', 'HTTPRequestHandler',
              'utf8Encoder', 'i', 'p_WindowNo', 'retValue', 'DB']
    for t in tests:
        print(t, '->', split_identifier(t))

    print()
    # 自测：结合stmt_block_builder跑一条真实样本
    from stmt_block_builder import build_stmt_blocks
    code = ('private int findPLV(int M_PriceList_ID){\n'
            '  Timestamp priceDate=null;\n'
            '  String dateStr=Env.getContext(Env.getCtx(),p_WindowNo,"DateOrdered");\n'
            '  int retValue=0;\n'
            '  return retValue;\n'
            '}\n')
    code_tokens, token_types, token2stmt, stmt_infos, block_infos = build_stmt_blocks(code)
    subtokens, sub2stmt = subtokenize_code(code_tokens, token_types, token2stmt)
    print('原始token数:', len(code_tokens), '-> 子词数:', len(subtokens))
    print(' '.join(subtokens))

    print()
    print('文本分词:', tokenize_text(' Find Price List Version and update context'))