# coding=utf-8
"""
对javalang的一层薄封装：输入原始数据集里的单个方法code字符串，
输出 (method_node, tokens)：
  - method_node: javalang解析出的MethodDeclaration节点(结构信息)
  - tokens: 按源码顺序排列的token列表，每个token带position(line,column)

数据集里的code字符串是"裸"的方法体(没有外层class)，例如：
    "@Override public int runCommand(...) throws IOException {\n  return ...;\n}\n"
javalang.parse.parse()要求输入是一个完整的compilation unit，所以这里和之前Java
tree-sitter版本一样的思路：包一层 "class _dummy_ { ... }" 再解析，解析完之后
所有token的position都是"包了壳之后"的坐标，之后统一按"跳过壳token"处理即可，
不需要重新计算相对位置(这一点比之前_truncate_ast_by_code_and_layout里的位置重算简单)。
"""
import javalang
from javalang.tokenizer import tokenize as javalang_tokenize

WRAPPER_PREFIX = 'class _dummy_ { '
WRAPPER_SUFFIX = ' }'


class JavaParseError(Exception):
    pass


def wrap_method_code(raw_code: str) -> str:
    """给方法体套一层class壳，使其成为javalang可以parse的合法compilation unit"""
    return WRAPPER_PREFIX + raw_code.strip() + WRAPPER_SUFFIX


def parse_method(raw_code: str):
    """
    返回 (method_node, tokens)
    method_node: javalang.tree.MethodDeclaration 或 ConstructorDeclaration
    tokens: list[javalang.tokenizer.JavaToken]，源码顺序，每个token.position=(line,column)

    注意：wrap之后的壳token(class/_dummy_/{/最后的})不会被剔除，
    调用方(stmt_block_builder.py)通过method_node本身的position范围来限定处理范围，
    天然就不会碰到壳token，不需要额外裁剪。
    """
    wrapped = wrap_method_code(raw_code)
    try:
        tokens = list(javalang_tokenize(wrapped))
    except Exception as e:
        raise JavaParseError('tokenize failed: {}'.format(e))

    try:
        tree = javalang.parse.parse(wrapped)
    except Exception as e:
        raise JavaParseError('parse failed: {}'.format(e))

    class_decl = tree.types[0]
    # 方法体里可能是MethodDeclaration，也可能是ConstructorDeclaration(极少数据集里会有构造函数样本)
    method_like = [m for m in class_decl.body
                   if type(m).__name__ in ('MethodDeclaration', 'ConstructorDeclaration')]
    if not method_like:
        raise JavaParseError('no method/constructor declaration found in wrapped code')
    method_node = method_like[0]
    return method_node, tokens


if __name__ == '__main__':
    sample = ('@Override public int runCommand(boolean mergeErrorIntoOutput,String... commands) '
               'throws IOException, InterruptedException {\n'
               '  return runCommand(mergeErrorIntoOutput,new ArrayList<String>(Arrays.asList(commands)));\n'
               '}\n')
    node, toks = parse_method(sample)
    print('method name:', node.name)
    print('body stmt count:', len(node.body) if node.body else 0)
    print('first 10 tokens:', [(t.value, t.position) for t in toks[:10]])