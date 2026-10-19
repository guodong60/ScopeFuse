#coding=utf-8
import ast
import numpy as np
from my_lib.util.nl_parser.en_parser import punc_str
from .code_tokenizer import tokenize_code_str
from tree_sitter import Language, Parser,Node
import os

def tokenize_python(code,lower=False,keep_punc=True,lemmatize=True,punc_str=punc_str,user_words=None,pos_tag=False,
                    seg_var=True,rev_dic=None):
    if user_words is None:
        user_words=[]
    operators=[]
    return tokenize_code_str(code=code,lower=lower,keep_punc=keep_punc,lemmatize=lemmatize,punc_str=punc_str,
                         user_words=user_words,operators=operators,pos_tag=pos_tag,seg_var=seg_var,rev_dic=rev_dic)

def py2ast(code,attr='all',seg_attr=True,lemmatize=True,lower=True, keep_punc=True,seg_var=True,rev_dic=None,user_words=None):
    '''
    :param code: str
    :param attr: str. Need attribute nodes or not. If not, the following parameters will be invalid
    :param seg_attr: bool. Segment the attribute nodes or not. If not, the following parameters will be invalid
    :return: [list,np.array,List[Tuple[int]]]
    '''

    def _get_children(node):
        # attr_chidren,func_children = [],[]
        # children = []

        for _, child in ast.iter_fields(node):
            if child and not isinstance(child,list):  # 如果是节点就直接添加
                # children.append(child)
                yield child
            elif isinstance(child, list):  # 如果是list或者tuple
                while len(child) == 1 and isinstance(child[0], list):
                    child = child[0]
                # children.extend(child)
                for item in child:
                    if item:
                        yield item
        # return children

    def _get_func_children(node):
        children = _get_children(node)
        func_children = filter(_is_func_node, children)
        return func_children
        # return ast.iter_child_nodes(node)

    def _get_attr_children(node):
        children = _get_children(node)
        attr_children = filter(_is_attr_node, children)
        return attr_children

    def _is_func_node(node):
        return node and isinstance(node, ast.AST)

    def _is_attr_node(node):
        return node and not isinstance(node, ast.AST)

    def _walk(node):  # 广度优先遍历所有功能节点
        """
        Recursively yield all descendant nodes in the tree starting at *node*
        (including *node* itself), in no specified order.  This is useful if you
        only want to modify nodes in place and don't care about the context.
        """
        from collections import deque
        todo = deque([node])
        while todo:
            node = todo.popleft()
            todo.extend(_get_func_children(node))
            yield node

    expr_ast = ast.parse(code)
    edge_end_ids = []
    edge_start_ids = []
    nodes = []
    depths = [0]
    subtree_poses = [0]
    subling_poses = [0]

    edge_start_id = 0
    # attr_edge_start_id = 1
    edge_end_id_queue = [0]

    edge_depth_queue = [1] * len(list(_get_func_children(expr_ast)))  # 边的深度的队列
    node_depth_queue = [0] * 1
    subtree_pos = -1
    attr_depth = -1

    str_node = ast.dump(expr_ast)
    str_node = str_node[:str_node.index('(')]
    nodes.append(str_node)
    for node in _walk(expr_ast):
        edge_end_id = edge_end_id_queue.pop(0)
        # assert isinstance(node,javalang.ast.Node)
        # str_node = str(node)
        # str_node = ast.dump(node)
        # str_node = str_node[:str_node.index('(')]
        # nodes.append(str_node)

        subtree_pos += 1
        if node_depth_queue[0] > attr_depth:
            subtree_pos = 0
        attr_depth = node_depth_queue.pop(0)
        # child_node_num =len(get_node_children(node))

        node_depth_queue.extend([attr_depth + 1] * len(list(_get_func_children(node))))  # 为什么max？

        children = _get_children(node)
        attr_subling_pos, func_subling_pos = 0, 0
        for child in children:
            if attr is not None and _is_attr_node(child):
                if (attr == 'str' and isinstance(child, str)) or attr == 'all':
                    if not repr(child).isdigit() and seg_attr:
                        #字符串repr后前后会带引号，需要用去掉
                        tokens = tokenize_python(repr(child).strip('"').strip("'").strip('"'),lemmatize=lemmatize,lower=lower, keep_punc=keep_punc,
                                                 seg_var=seg_var,rev_dic=rev_dic,user_words=user_words)
                        for j, token in enumerate(tokens):
                            edge_end_ids.append(edge_end_id)
                            edge_start_id += 1
                            edge_start_ids.append(edge_start_id)
                            nodes.append(token)

                            depths.append(attr_depth + 1)
                            subtree_poses.append(subtree_pos)
                            subling_poses.append(-(attr_subling_pos + 1 + j))
                    else:
                        edge_end_ids.append(edge_end_id)
                        edge_start_id += 1
                        edge_start_ids.append(edge_start_id)
                        nodes.append(repr(child).strip('"').strip("'").strip('"'))

                        depths.append(attr_depth + 1)
                        subtree_poses.append(subtree_pos)
                        subling_poses.append(-(attr_subling_pos + 1))
                    attr_subling_pos += 1
            elif _is_func_node(child):
                str_node = ast.dump(child)
                str_node = str_node[:str_node.index('(')]
                nodes.append(str_node)

                edge_end_ids.append(edge_end_id)
                edge_start_id += 1
                edge_start_ids.append(edge_start_id)
                edge_end_id_queue.append(edge_start_id)

                depths.append(edge_depth_queue.pop(0))
                edge_depth_queue.extend(
                    [depths[-1] + 1] * len(list(_get_func_children(child))))  ##为什么max？ 如果当前点为树，则将其下所有边的深度值加入深度值队列
                subtree_poses.append(subtree_pos)
                subling_poses.append(func_subling_pos)
                func_subling_pos += 1

    edges = np.array([edge_start_ids, edge_end_ids])    #这里的边是从子节点到父节点的
    node_poses = list(zip(depths, subtree_poses, subling_poses))
    return nodes, edges, node_poses


def py2ast_sitter(code,attr='all',seg_attr=True,lemmatize=True,lower=True, keep_punc=True,seg_var=True,rev_dic=None,user_words=None):
    '''
    :param code: str
    :param attr: str. Need attribute nodes or not. If not, the following parameters will be invalid
    :param seg_attr: bool. Segment the attribute nodes or not. If not, the following parameters will be invalid
    :return: [list,np.array,List[Tuple[int]]]
    '''

    def _get_children(node):
        if node.child_count>0:
            func_children=list(filter(_is_func_node, node.children))
            if len(func_children)==0:    #如果children是个空序列
                return [str(node.text,encoding='utf-8')]
            return func_children
        if node.is_named and str(node.text, encoding='utf-8').lower() != node.type.lower():   #前提是已经过滤掉那些没用的unnamed
            return [str(node.text,encoding='utf-8')]
        return []

    def _get_func_children(node):
        children = _get_children(node)
        func_children = filter(_is_func_node, children)
        return func_children
        # return ast.iter_child_nodes(node)

    def _get_attr_children(node):
        children = _get_children(node)
        attr_children = filter(_is_attr_node, children)
        return attr_children

    def _to_str(node):
        if _is_attr_node(node):
            return node
        elif _is_func_node(node):
            return node.type

    def _is_func_node(node):    #如果有named子节点，说明是功能节点
        if isinstance(node,str):
            return False
        if node.type=='comment':
            return False
        if not node.is_named and \
            not node.parent.type.endswith('assignment') and \
            not node.parent.type.endswith('operator') and \
            not node.parent.type.endswith('modifiers'):
            # 该条件处后面可能还需要根据需求来改，第一个对应= +=等赋值操作，第二个对应+，-等操作符，第三个对应public private这类（python 貌似没有)
            return False
        return True

    def _is_attr_node(node):    #如果有named子节点或者type
        if isinstance(node,str):
            return True
        return False

    def _walk(node):  # 广度优先遍历所有功能节点
        """
        Recursively yield all descendant nodes in the tree starting at *node*
        (including *node* itself), in no specified order.  This is useful if you
        only want to modify nodes in place and don't care about the context.
        """
        from collections import deque
        todo = deque([node])
        while todo:
            node = todo.popleft()
            todo.extend(_get_func_children(node))
            yield node

    # from .tree_sitter_repo import my
    cur_dir=os.path.dirname(os.path.abspath(__file__))
    lan_path=os.path.join(cur_dir,'tree_sitter_repo/my-languages.so')
    py_language = Language(lan_path, 'python')
    parser = Parser()
    parser.set_language(py_language)
    code = bytes(code, 'utf8') #caution
    tree = parser.parse(code)
    root_node=tree.root_node

    # expr_ast = ast.parse(code)
    edge_end_ids = []
    edge_start_ids = []
    nodes = []
    depths = [0]
    subtree_poses = [0]
    subling_poses = [0]

    edge_start_id = 0
    # attr_edge_start_id = 1
    edge_end_id_queue = [0]

    edge_depth_queue = [1] * len(list(_get_func_children(root_node)))  # 边的深度的队列
    node_depth_queue = [0] * 1
    subtree_pos = -1
    attr_depth = -1

    str_node = root_node.type #caution
    nodes.append(str_node)
    for node in _walk(root_node):
        edge_end_id = edge_end_id_queue.pop(0)

        subtree_pos += 1
        if node_depth_queue[0] > attr_depth:
            subtree_pos = 0
        attr_depth = node_depth_queue.pop(0)
        # str_node=root_node.type
        # child_node_num =len(get_node_children(node))

        node_depth_queue.extend([attr_depth + 1] * len(list(_get_func_children(node))))  # 为什么max？

        children = _get_children(node)
        # str_children=[_to_str(child) for child in children]
        attr_subling_pos, func_subling_pos = 0, 0
        for child in children:
            if attr is not None and _is_attr_node(child):
                if (attr == 'str' and node.type=='string') or attr == 'all':
                    if not repr(child).isdigit() and seg_attr:
                        #字符串repr后前后会带引号，需要用去掉
                        tokens = tokenize_python(repr(child).strip('"').strip("'").strip('"'),lemmatize=lemmatize,lower=lower, keep_punc=keep_punc,
                                                 seg_var=seg_var,rev_dic=rev_dic,user_words=user_words)
                        for j, token in enumerate(tokens):
                            edge_end_ids.append(edge_end_id)
                            edge_start_id += 1
                            edge_start_ids.append(edge_start_id)
                            nodes.append(token)

                            depths.append(attr_depth + 1)
                            subtree_poses.append(subtree_pos)
                            subling_poses.append(-(attr_subling_pos + 1 + j))
                    else:
                        edge_end_ids.append(edge_end_id)
                        edge_start_id += 1
                        edge_start_ids.append(edge_start_id)
                        nodes.append(repr(child).strip('"').strip("'").strip('"'))

                        depths.append(attr_depth + 1)
                        subtree_poses.append(subtree_pos)
                        subling_poses.append(-(attr_subling_pos + 1))
                    attr_subling_pos += 1
            elif _is_func_node(child):
                # str_node = ast.dump(child)
                # str_node = str_node[:str_node.index('(')]
                str_node=child.type
                nodes.append(str_node)

                edge_end_ids.append(edge_end_id)
                edge_start_id += 1
                edge_start_ids.append(edge_start_id)
                edge_end_id_queue.append(edge_start_id)

                depths.append(edge_depth_queue.pop(0))
                edge_depth_queue.extend(
                    [depths[-1] + 1] * len(list(_get_func_children(child))))  ##为什么max？ 如果当前点为树，则将其下所有边的深度值加入深度值队列
                subtree_poses.append(subtree_pos)
                subling_poses.append(func_subling_pos)
                func_subling_pos += 1

    edges = np.array([edge_start_ids, edge_end_ids])    #这里的边是从子节点到父节点的
    node_poses = list(zip(depths, subtree_poses, subling_poses))
    return nodes, edges, node_poses

if __name__=='__main__':
    import astunparse

    # from my_tokenizer import tokenize_python, tokenize_java
    code = '''
a='This is a test'
        '''
    # ast_exp= ast.parse(code)
    # print(astunparse.dump(ast_exp))

    nodes, edges, poses = py2ast_sitter(code, attr='all', seg_attr=True, lemmatize=True, lower=True, keep_punc=True,
                                 seg_var=True, )
    print(list(zip(nodes, list(range(len(nodes))), poses, )))
    print(edges)
    print(poses)

    nodes, edges, poses = py2ast(code,attr='all',seg_attr=True,lemmatize=True,lower=True,keep_punc=True,seg_var=True,)
    print(list(zip(nodes,list(range(len(nodes))),poses,)))
    print(edges)
    print(poses)



    # def walk(node):  # 广度优先遍历所有功能节点
    #     """
    #     Recursively yield all descendant nodes in the tree starting at *node*
    #     (including *node* itself), in no specified order.  This is useful if you
    #     only want to modify nodes in place and don't care about the context.
    #     """
    #     from collections import deque
    #     todo = deque([node])
    #     while todo:
    #         node = todo.popleft()
    #         todo.extend(node.children)
    #         yield node
    #
    # def is_func_node(node):    #如果有named子节点，说明是功能节点
    #     if isinstance(node,str):
    #         return False
    #     if node.type=='comment':
    #         return False
    #     if not node.is_named and not (node.parent.type.endswith('assignment') or node.parent.type.endswith('operator')):
    #         return False
    #     return True

    # from tree_sitter import Language, Parser
    #
    # # from .tree_sitter_repo import my
    # py_language = Language('tree_sitter_repo/my-languages.so', 'python')
    # parser = Parser()
    # parser.set_language(py_language)
    # bcode = bytes(code, 'utf8')
    # tree = parser.parse(bcode)
    # # print(tree.root_node.children)
    #
    # i = 0
    # for child in walk(tree.root_node):
    #     # if is_func_node(child):
    #     print(child)
    #     print(str(child.text,encoding="utf-8"))
    #     print(child.type)
    #     print(child.is_named)
    #     print("***" * 20)
    #     i += 1
    # print(i)
