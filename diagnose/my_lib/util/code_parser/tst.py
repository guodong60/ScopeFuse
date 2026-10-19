#coding=utf-8


if __name__=='__main__':
    code = '''
while False:
 c+=b-a
 d=c*3
'''
#     code='''
# class _ {
#     String s="a Test";
#     Integer a=1;
#     int b=1;
#     int c;
#     c=a+b;
# }'''
    # nodes, edges, poses = java2ast_sitter(code,attr='all',seg_attr=True,lemmatize=True,lower=True,keep_punc=True,seg_var=True,)
    # print(list(zip(nodes,list(range(len(nodes))),poses,)))
    # print(edges)
    # print(poses)
    # print('***'*20)
    # nodes, edges, poses = java2ast(code, attr='all', seg_attr=True, lemmatize=True, lower=True, keep_punc=True,
    #                                       seg_var=True, )
    # print(list(zip(nodes, list(range(len(nodes))), poses, )))
    # print(edges)
    # print(poses)

    def walk(node):  # 广度优先遍历所有功能节点
        """
        Recursively yield all descendant nodes in the tree starting at *node*
        (including *node* itself), in no specified order.  This is useful if you
        only want to modify nodes in place and don't care about the context.
        """
        from collections import deque
        todo = deque([node])
        while todo:
            node = todo.popleft()
            todo.extend(node.children)
            yield node

    def is_func_node(node):    #如果有named子节点，说明是功能节点
        if isinstance(node,str):
            return False
        if node.type=='comment':
            return False
        if not node.is_named and not (node.parent.type.endswith('assignment') or node.parent.type.endswith('operator')):
            return False
        return True

    from tree_sitter import Language, Parser

    # from .tree_sitter_repo import my
    py_language = Language('tree_sitter_repo/my-languages.so', 'python')
    parser = Parser()
    parser.set_language(py_language)
    bcode = bytes(code, 'utf8')
    tree = parser.parse(bcode)
    # print(tree.root_node.children)

    i = 0
    print(tree.root_node)
    for child in walk(tree.root_node):
        # if child.type=="=":
        print(child)
        print(child.type)
        print(str(child.text,encoding="utf-8"))
        print(child.is_named)
        print("***" * 20)
        # print(child.parent.type,str(child.parent.text,encoding="utf-8"))
        i += 1
    print(i)