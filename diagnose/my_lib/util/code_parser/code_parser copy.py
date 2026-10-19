#coding=utf-8
'''
from https://github.com/tree-sitter/py-tree-sitter
'''
from copy import deepcopy
import numpy as np
from my_lib.util.nl_parser.en_parser import punc_str
from .code_tokenizer import tokenize_code_str
from tree_sitter import Language, Parser,Node
import os,re
from collections import deque
from ._DFG import *

class SitParser(object):
    def __init__(self,lan:str='python',lemmatize:bool=True,lower:bool=True,seg_attr=True, ast_intact:bool=True,
                 rev_dic=None,user_words=None):
        '''

        :param lan: 语言，目前测试了python和java版本良好
        :param lemmatize: 是否对变量字符串等attribute node做lemmatize
        :param lower: 是否将变量字符串等attribute node做小写转换
        :param ast_intact: 是否输出完整的AST，包括各种冗余的标点符号比如{ ; .等等
        :param rev_dic: revised dictionary
        :param user_words: 用户字典
        '''
        # self.attr=attr
        self.language=lan
        self.lemmatize=lemmatize
        self.lower=lower
        self.seg_attr=seg_attr
        self.ast_intact=ast_intact
        self.puncs=set() #标点符号
        # self.punc_str = '' if punc_str is None else punc_str
        self.operators = set()
        self.digits=set()  #数字
        self.user_words = [] if user_words is None else user_words
        self.user_words += [tuple(operator) for operator in self.operators]
        self.rev_dic=rev_dic

        cur_dir = os.path.dirname(os.path.abspath(__file__))
        lan_path = os.path.join(cur_dir, 'tree_sitter_repo/my-languages.so')
        py_language = Language(lan_path, lan)
        self.parser = Parser()
        self.parser.set_language(py_language)

    def _get_children(self,node):
        '''
        获取生成子节点列表
        :param node:
        :return:
        '''
        if node.child_count > 0:
            func_children = list(filter(self._is_func_node, node.children))
            if len(func_children) == 0:  # 如果children是个空序列,针对string类型的node
                return [str(node.text, encoding='utf-8')]
            return func_children
        # 针对一些没有func子节点的func_node，如果type和text不同才将起text判定为其attr_node，否则判定其没有attr子节点
        if node.is_named and str(node.text, encoding='utf-8').lower() != node.type.lower():  # 前提是已经过滤掉那些没用的unnamed
            return [str(node.text, encoding='utf-8')]
        return []

    def _get_func_children(self,node):
        '''
        获取功能子节点列表
        :param node:
        :return:
        '''
        children = self._get_children(node)
        func_children = filter(self._is_func_node, children)
        return func_children
        # return ast.iter_child_nodes(node)

    def _get_attr_children(self,node):
        '''
        获取属性子节点列表
        :param node:
        :return:
        '''
        children = self._get_children(node)
        attr_children = filter(self._is_attr_node, children)
        return attr_children

    def _to_str(self,node):
        '''
        提取节点名称字符串
        :param node:
        :return:
        '''
        if self._is_attr_node(node):
            if isinstance(node, str):
                return node
            if node.type=='escape_sequence':    #a=='daa'这种string下还会有一个escape_sequence然后才接着'daa'，将escape_sequence节点视为attr node,前后的引号已经被隔开了，这里要加上，形成和string节点中内容一样的形式
                return str(node.prev_sibling.text, encoding='utf-8')+str(node.text,encoding='utf-8')+str(node.next_sibling.text, encoding='utf-8')
        elif self._is_func_node(node):
            return node.type

    def _is_func_node(self,node):
        '''
        是否为功能节点
        :param node:
        :return:
        '''
        if isinstance(node, str):
            return False
        # if node.type == 'comment':
        if 'comment' in node.type:  #comment (python) or line_comment (java)
            return False
        # if node.start_point==node.end_point:
        #     #代码无法完全解析有错误ERROR节点时，可能出现一些;符号，此时node.start_point==node.end_point
        #     return False
        if node.type=='"':  #python里字符串节点(string)有两个子节点都是'"'，不需要，可以丢掉，因为text字符串中也有引号
            return False
        if node.type=='escape_sequence':    #a=='daa'这种string下还会有一个escape_sequence然后才接着'daa'，将escape_sequence节点视为attr node
            return False
        if not self.ast_intact and \
            not node.is_named and \
            node.parent.named_child_count>0 and \
            not node.parent.type.endswith('assignment') and \
            not node.parent.type.endswith('operator') and \
            not node.parent.type.endswith('modifiers'): # 如果有named子节点，说明是功能节点
            # 避免操作符+=等被不判定为func_node,该条件处后面可能还需要根据需求来改，第一个对应= +=等赋值操作，第二个对应+，-等操作符，第三个对应public private这类，
            return False

        return True

    def _is_redundant_func_node(self,node):
        '''
        是否为冗余的功能节点，比如 ; } 这些符号或者 while for这些关键字，这些关键字在父节点中有体现比如while_statement
        :param node:
        :return:
        '''
        if self._is_func_node(node) and not node.is_named and node.parent.named_child_count>0 and \
            not node.parent.type.endswith('assignment') and \
            not node.parent.type.endswith('operator') and \
            not node.parent.type.endswith('modifiers'): # 如果有named子节点，说明是功能节点
            # 避免操作符+=等被不判定为func_node,该条件处后面可能还需要根据需求来改，第一个对应= +=等赋值操作，第二个对应+，-等操作符，第三个对应public private这类，
            return True
        return False

    def _is_attr_node(self,node):  # 如果有named子节点或者type
        '''
        是否为属性节点
        :param node:
        :return:
        '''
        if isinstance(node, str):
            return True
        if node.type=='escape_sequence':    #a=='daa'这种string下还会有一个escape_sequence然后才接着'daa'，将escape_sequence节点视为attr node
            return True
        return False

    def _is_operator_func_node(self,node):
        '''
        判断该功能节点是不是为+ - +=等这些操作符,但是比如python中的in or and这些操作符除去
        比如:
        d+=a+b中的
        <Node kind="+=", start_point=(4, 2), end_point=(4, 4)>
        +=
        +=
        False
        '''
        if not node.is_named and (node.parent.type.endswith('assignment') or node.parent.type.endswith('operator')):
            if re.search(r'[A-Za-z0-9]',str(node.text, encoding='utf-8'),flags=re.S) is None:
                return True
        return False

    def _is_digit_func_node(self,node):
        '''
        判断该功能节点是不是数字,node.type中带有integer或者float的认为是,
        但比如java中变量定义出现double等也会因为type被定义为数字,要去除
        :param node:
        :return:
        '''
        if re.search(r'integer|float', node.type, flags=re.S) is not None:
            if re.search(r'[0-9]',str(node.text, encoding='utf-8'),flags=re.S) is not None:
                return True
        return False

    def _walk(self,node):  # 广度优先遍历所有功能节点
        """
        Recursively yield all descendant nodes in the tree starting at *node*
        (including *node* itself), in no specified order.  This is useful if you
        only want to modify nodes in place and don't care about the context.
        """
        todo = deque([node])
        while todo:
            node = todo.popleft()
            todo.extend(self._get_func_children(node))
            yield node

    def _get_ast_info(self,tree):
        '''
        生成AST信息，包括nodes，edges，positions
        :param tree:
        :return:
        '''
        # nodes, edges, node_poses, node_points= [], np.array([]), [], [],[]
        # byte_code = bytes(code, 'utf8')  # caution
        # tree = self.parser.parse(byte_code)
        root_node = tree.root_node
        #
        # # 找出该段代码中的所有操作符operators和不包括操作符的标点符号(puncs)
        # self.puncs |= set(re.findall(r'[^0-9A-Za-z\u4e00-\u9fa5]', code, flags=re.S))
        # for node in self._walk(root_node):
        #     if not node.is_named and (node.parent.type.endswith('assignment') or node.parent.type.endswith('operator')):
        #         self.operators.add(str(node.text, encoding='utf-8'))
        #     # if node.type in ['float','integer']:
        #     if re.search(r'integer|float', node.type, flags=re.S) is not None:
        #         self.digits.add(str(node.text, encoding='utf-8'))
        # self.puncs -= self.operators

        edge_end_ids, edge_start_ids = [], []
        nodes, node_points = [], []
        depths, subtree_poses, sibling_poses = [0], [0], [0]

        edge_start_id = 0
        edge_end_id_queue = [0]

        edge_depth_queue = [1] * len(list(self._get_func_children(root_node)))  # 边的深度的队列
        node_depth_queue = [0] * 1
        subtree_pos = -1
        attr_depth = -1

        str_node = root_node.type  # caution

        # self._source_code = root_node.text

        nodes.append(str_node)
        node_points.append((root_node.start_point,root_node.end_point))
        if self.ast_intact:
            redundant_node_tags=[False]
        for node in self._walk(root_node):
            # if node.type in ['"',"'"]:
            #     print()
            edge_end_id = edge_end_id_queue.pop(0)

            subtree_pos += 1
            if node_depth_queue[0] > attr_depth:
                subtree_pos = 0
            attr_depth = node_depth_queue.pop(0)
            # child_node_num =len(get_node_children(node))

            node_depth_queue.extend([attr_depth + 1] * len(list(self._get_func_children(node))))  # 为什么max？
            children = self._get_children(node)
            # if node.type=='modifiers':
            #     print([_to_str(child) for child in _get_children(node)])
            #     print()
            attr_sibling_pos, func_sibling_pos = 0, 0
            for child in children:
                if self._is_attr_node(child):
                    if isinstance(child, str):
                        str_node=child
                    elif node.type=='escape_sequence':    #a=='daa'这种string下还会有一个escape_sequence然后才接着'daa'，将escape_sequence节点视为attr node,前后的引号已经被隔开了，这里要加上，形成和string节点中内容一样的形式
                        str_node=str(child.prev_sibling.text, encoding='utf-8')+str(child.text,encoding='utf-8')+str(child.next_sibling.text, encoding='utf-8')
                    # str_node=self._to_str(child)
                    if self.seg_attr and str_node not in self.digits:  # 如果不是数字
                        tokens = tokenize_code_str(str_node,   #.strip('"').strip("'").strip('"')
                                                   lemmatize=self.lemmatize, lower=self.lower,
                                                   keep_punc=True,  # Note,这里必须统一是True或False，不然后面ast_intact=False时会不一致
                                                   rev_dic=self.rev_dic, user_words=self.user_words,
                                                   punc_str=''.join(self.puncs), operators=self.operators,
                                                   pos_tag=False)
                    else:
                        tokens = [str_node]    #.strip('"').strip("'").strip('"')

                    for j, token in enumerate(tokens):
                        # if token in ['"', "'"]:
                        #     print()
                        edge_end_ids.append(edge_end_id)
                        edge_start_id += 1
                        edge_start_ids.append(edge_start_id)
                        nodes.append(token)

                        depths.append(attr_depth + 1)
                        subtree_poses.append(subtree_pos)
                        sibling_poses.append(-(attr_sibling_pos + 1 + j))
                        node_points.append(())  #属性节点不加points
                        if self.ast_intact:
                            redundant_node_tags.append(False)
                    attr_sibling_pos += 1

                elif self._is_func_node(child):
                    # if self._is_operator_func_node(child):  #如果是操作符，加入操作符集合用于分词判定
                    #     self.operators.add(child.text)
                    str_node = child.type  # True会变成true，False会变成false
                    nodes.append(str_node)
                    # if str_node in ['"',"'"]:
                    #     print(self.ast_intact)
                    #     print(self._is_func_node(child))
                    #     print(self._is_redundant_func_node(child))
                    #     print(child.type)
                    #     print(str(child.text, encoding="utf-8"))
                    #     print(child.is_named)
                    #     print(child.parent.type)
                    #     print(child.parent.named_child_count,child.parent.child_count)
                    #     print()

                    node_points.append((child.start_point,child.end_point))
                    if self.ast_intact:
                        if self._is_redundant_func_node(child): #判断该function node 是否为冗余节点
                            redundant_node_tags.append(True)
                        else:
                            redundant_node_tags.append(False)
                    edge_end_ids.append(edge_end_id)
                    edge_start_id += 1
                    edge_start_ids.append(edge_start_id)
                    edge_end_id_queue.append(edge_start_id)

                    depths.append(edge_depth_queue.pop(0))
                    edge_depth_queue.extend([depths[-1] + 1] * len(list(self._get_func_children(child))))  ##为什么max？ 如果当前点为树，则将其下所有边的深度值加入深度值队列
                    subtree_poses.append(subtree_pos)
                    sibling_poses.append(func_sibling_pos)
                    func_sibling_pos += 1

        edges = np.array([edge_start_ids, edge_end_ids])  # 这里的边是从子节点到父节点的
        node_poses = list(zip(depths, subtree_poses, sibling_poses))
        if self.ast_intact:
            return nodes, edges, node_poses, node_points,redundant_node_tags
        else:
            return nodes, edges, node_poses, node_points

    def _pre_DFS_ids(self,edges):   #前序深度优先遍历
        '''
        根据边edges二维numpy数组前序深度优先遍历节点在节点序列中的序号
        :param edges:
        :return:
        '''
        assert edges[0,0]>edges[1,0]
        # if edges[0,0]<edges[1,0]:
        #     edges=np.array([edges[1,:],edges[0,:]])
        # root_id=edges[1,0]
        node_id_stack=[edges[1,0]]
        while node_id_stack:
            cur_node_id = node_id_stack.pop(-1)
            edge_ids = np.argwhere(edges[1, :] == cur_node_id)
            child_ids = [edges[0, idx[0]] for idx in edge_ids]
            node_id_stack.extend(reversed(child_ids))
            yield cur_node_id

    def _get_intact_ast_node_in_code_poses(self,intact_ast_edges,intact_ast_node_points):
        '''
        获取完整AST的所有节点在源代码中的对应位置，功能节点的位置以对应首源码位置记录
        :param intact_ast_edges:
        :param intact_ast_node_points:
        :return:
        '''
        node_in_code_poses = [tuple()] * len(intact_ast_node_points)
        line2indent=dict()
        for node_point in intact_ast_node_points:
            if node_point and (node_point[0][0] not in line2indent.keys() or line2indent[node_point[0][0]]>node_point[0][1]):
                line2indent[node_point[0][0]]=node_point[0][1]
        indent2offset=dict([(indent,offset) for offset,indent in enumerate(sorted(set(line2indent.values())))])
        line2spos=dict([(line,indent2offset[indent]) for line,indent in line2indent.items()])   #每行对应的起始位置
        node_in_code_poses = [tuple()] * len(intact_ast_node_points)
        cur_line = -1
        # start_pos = -1
        cur_attr_node_flag = True
        last_node_id = -1
        # last_indent = -1
        for node_id in self._pre_DFS_ids(intact_ast_edges):
            if intact_ast_node_points[node_id]:
                cur_attr_node_flag = False
                if intact_ast_node_points[node_id][0][0] > cur_line:  # 行号从0开始的,如果是新的下一行
                    # cur_line = intact_ast_node_points[node_id][0][0]  # 更新行号
                    # if intact_ast_node_points[node_id][0][1] > last_indent:  # 如果新行缩进大于上一行
                    #     start_pos += 1  # 更新本行缩进值,+1
                    # elif intact_ast_node_points[node_id][0][1] < last_indent:  # 如果新行缩进小于上一行:
                    #     start_pos -= 1  # 更新本行缩进值,-1
                    # print(intact_ast_node_points[node_id][0][1])
                    # assert start_pos >= 0  # 保证缩进值不小于0
                    # last_indent = intact_ast_node_points[node_id][0][1]
                    # # node_in_code_poses.append((cur_line, start_pos))
                    # node_in_code_poses[node_id] = (cur_line, start_pos)

                    cur_line = intact_ast_node_points[node_id][0][0]
                    node_in_code_poses[node_id]=(cur_line,line2spos[cur_line])
                else:
                    offset = 0 if intact_ast_node_points[last_node_id] and \
                                  intact_ast_node_points[node_id][0][1] == intact_ast_node_points[last_node_id][0][1] else 1
                    node_in_code_poses[node_id] = (cur_line, node_in_code_poses[last_node_id][-1] + offset)
            else:
                offset = 0 if not cur_attr_node_flag else 1
                node_in_code_poses[node_id] = (cur_line, node_in_code_poses[last_node_id][-1] + offset)
                cur_attr_node_flag = True
            last_node_id = node_id

        return node_in_code_poses

    def parse(self,code):
        '''
        解析代码，包括解析出格式化后的源码source_code，完整AST信息（nodes,edges,positions)，AST节点在source code中的位置
        如果ast_intact=False，还要解析精简AST的所有信息
        :param code:
        :return:
        '''
        byte_code = bytes(code, 'utf8')  # caution
        self._tree = self.parser.parse(byte_code)
        root_node = self._tree.root_node

        # 规整后的源代码，通过这种方式下述代码可以合成一行
        # String
        # s = \
        #     "";
        self._source_code = str(root_node.text, encoding='utf-8')

        # 找出该段代码中的所有操作符operators和不包括操作符的标点符号(puncs)
        self.puncs |= set(re.findall(r'[^0-9A-Za-z\u4e00-\u9fa5\n \t]', code, flags=re.S))
        for node in self._walk(root_node):
            # if not node.is_named and (node.parent.type.endswith('assignment') or node.parent.type.endswith('operator')):
            if self._is_operator_func_node(node):
                self.operators.add(str(node.text, encoding='utf-8'))
            # if node.type in ['float','integer']:
            # if re.search(r'integer|float', node.type, flags=re.S) is not None:
            if self._is_digit_func_node(node):
                self.digits.add(str(node.text, encoding='utf-8'))
        self.puncs -= self.operators

        #First, we must make sure to get a whole ast with redundant nodes
        ast_intact=self.ast_intact
        self.ast_intact=True
        self._intact_ast_nodes,self._intact_ast_edges,self._intact_ast_node_poses,\
        self._intact_ast_node_points,self._intact_redundant_ast_node_tags=self._get_ast_info(tree=self._tree)
        self._intact_ast_node_in_code_poses=self._get_intact_ast_node_in_code_poses(intact_ast_edges=self._intact_ast_edges,
                                                                                    intact_ast_node_points=self._intact_ast_node_points)

        #Then if ast_intact is True, generate the concise AST information
        self.ast_intact=ast_intact
        if not self.ast_intact:
            self._concise_ast_nodes, self._concise_ast_edges, self._concise_ast_node_poses, \
            self._concise_ast_node_points = self._get_ast_info(tree=self._tree)
            self._concise_ast_node_in_code_poses=[pos for pos,tag in zip(self._intact_ast_node_in_code_poses,
                                                                         self._intact_redundant_ast_node_tags) if not tag]
            # red_nodes=[node for node,tag in zip(self._intact_ast_nodes,self._intact_redundant_ast_node_tags) if tag]
            # print(len(red_nodes),red_nodes)
            # print(len(self._concise_ast_node_in_code_poses),len(self._concise_ast_nodes))
            assert len(self._concise_ast_node_in_code_poses)==len(self._concise_ast_nodes)

    @property
    def source_code(self):
        '''
        格式化后的源代码，有些原始代码出现中间换行的情况可以避免
        :return:
        '''
        return self._source_code

    @property
    def code_tokens(self):
        '''
        通过从左至右从上到下的前序深度优先遍历源代码得到代码token序列,注意是完整的code，不是concise AST的code
        :return:list
        '''
        code_tokens = []
        for node_id in self._pre_DFS_ids(self._intact_ast_edges):
            if node_id not in self._intact_ast_edges[1, :]:
                code_tokens.append(self._intact_ast_nodes[node_id])
        return code_tokens

    @property
    def code_token_poses(self):
        '''
        代码token的位置序列
        :return:
        '''
        code_token_poses = []
        for node_id in self._pre_DFS_ids(self._intact_ast_edges):
            if node_id not in self._intact_ast_edges[1, :]:
                code_token_poses.append(self._intact_ast_node_in_code_poses[node_id])
        return code_token_poses

    @property
    def ast_nodes(self):    #ast node
        '''
        AST node序列
        :return:
        '''
        if self.ast_intact:
            return self._intact_ast_nodes
        return self._concise_ast_nodes

    @property
    def ast_edges(self):    #ast edges
        '''
        AST edge二维numpy数组[2,edge_num],从下往上连接,[0,:]为子节点，[1,:]为父节点
        :return:
        '''
        if self.ast_intact:
            return self._intact_ast_edges
        return self._concise_ast_edges
    
    @property
    def ast_sibling_edges(self):
        '''
        从左至右的兄弟节点之间的边，[0,:]为左节点，[1,:]为右节点
        :return:
        '''
        sibling_edges=np.empty(shape=(2,0),dtype=np.int64)
        for father_id in sorted(set(self.ast_edges[1,:])):
            child_ids=sorted(self.ast_edges[0,:][np.argwhere(self.ast_edges[1,:]==father_id)].reshape((-1,)))
            if len(child_ids)>1:
                edges=np.array([child_ids[:-1],child_ids[1:]])
                sibling_edges=np.concatenate([sibling_edges,edges],axis=-1)
        return sibling_edges

    @property
    def ast_node_poses(self):    #ast positions
        '''
        AST node position序列
        :return:
        '''
        if self.ast_intact:
            return self._intact_ast_node_poses
        return self._concise_ast_node_poses

    @property
    def ast_node_in_code_poses(self):
        '''
        AST node在源码中的位置
        :return:
        '''
        # code_lines=self._source_code.split('\n')
        if self.ast_intact:
            return self._intact_ast_node_in_code_poses
        return self._concise_ast_node_in_code_poses

    @property
    def code_token_edges(self):
        '''
        源代码的文本形式在AST图中的连接边,从前往后连接
        :return:
        '''
        code_node_id_path = []
        for node_id in self._pre_DFS_ids(self.ast_edges):
            if node_id not in self.ast_edges[1, :]:
                code_node_id_path.append(node_id)
                # code_edges.append([node_id])
        return np.array([code_node_id_path[:-1],code_node_id_path[1:]])  # 第一组和最后一组要除去

    @property
    def code_layout_edges(self):
        '''
        源代码的布局形式在AST图中的连接边,从下往上连接,[0,:]为子节点，[1,:]为父节点
        布局图根据代码行以及缩进来形成图连接
        :return:
        '''
        code_node_ids=[]    #代码token的node id序列
        code_token_poses=self.code_token_poses  #如果是全树
        ast_edges=deepcopy(self.ast_edges)
        #首先找出源代码序列对应的AST节点id
        for node_id in self._pre_DFS_ids(ast_edges):
            if node_id not in ast_edges[1, :]:
                code_node_ids.append(node_id)
        if not self.ast_intact: #如果不是全树
            code_token_poses=[] #code_token_poses重新初始化
            for node_id in self._pre_DFS_ids(self._intact_ast_edges):   #前序深度优先遍历节点
                if node_id not in self._intact_ast_edges[1, :] and not self._intact_redundant_ast_node_tags[node_id]:
                    #如果该节点为叶节点并且不为冗余节点，则加入
                    code_token_poses.append(self._intact_ast_node_in_code_poses[node_id])
        assert len(code_node_ids)==len(code_token_poses)    #断言，节点id数量和代码token位置数量一样

        line_start_node_idss=[set()]*(code_token_poses[-1][0]+1)   #每行的起始的node id集合序列，下标代表行号
        line_start_pos=(0,0)    #初始化行起始位置
        for code_node_id,code_token_pos in zip(code_node_ids,code_token_poses):
            line_start_node_ids=set()
            if not line_start_node_idss[code_token_pos[0]]:
                line_start_pos=code_token_pos

            child_node_id=code_node_id
            while child_node_id in ast_edges[0,:]:
                father_node_id = ast_edges[1, :][ast_edges[0, :].tolist().index(child_node_id)]
                if self.ast_node_in_code_poses[father_node_id][0]==line_start_pos[0]:
                    if self.ast_node_in_code_poses[father_node_id][1]<=self.ast_node_in_code_poses[child_node_id][1]:
                        if self.ast_node_in_code_poses[father_node_id][1]<self.ast_node_in_code_poses[child_node_id][1]:
                            line_start_node_ids.discard(child_node_id)
                        line_start_node_ids.add(father_node_id)
                else:
                     break
                child_node_id=father_node_id

            if line_start_node_idss[code_token_pos[0]]:
                line_start_node_idss[code_token_pos[0]]&=line_start_node_ids
            else:
                line_start_node_idss[code_token_pos[0]]=line_start_node_ids
        line_start_node_ids=[max(line_start_node_id_set) if line_start_node_id_set else None
                             for line_start_node_id_set in line_start_node_idss]

        code_layout_edges=[[],[]]
        for line_start_node_id in line_start_node_ids:
            #先将所有line_start_node_id互相连通
            if line_start_node_id is not None:  #每个点只需要找到其layout图中的父节点即可，合成后必全部连通
                child_node_id=line_start_node_id
                while child_node_id in ast_edges[0,:]:
                    father_node_id = ast_edges[1, :][ast_edges[0, :].tolist().index(child_node_id)]
                    if father_node_id in line_start_node_ids:
                        code_layout_edges[0].append(line_start_node_id) #这里是line_start_node_id而不是child_node_id！
                        code_layout_edges[1].append(father_node_id)
                        break   #找到即停止
                    child_node_id=father_node_id
        line_start_root_node_ids=[idx for idx in line_start_node_ids if idx is not None and idx not in code_layout_edges[0]]
        if len(line_start_root_node_ids)>1: #如果根节点超过1个说明需要找一个公共祖父节点
            root_node_ids = set()
            for line_start_node_id in line_start_root_node_ids:
                tmp_root_node_ids = set()
                child_node_id=line_start_node_id
                while child_node_id in ast_edges[0,:]:
                    father_node_id=ast_edges[1,:][ast_edges[0,:].tolist().index(child_node_id)]
                    tmp_root_node_ids.add(father_node_id)
                    child_node_id=father_node_id
                if root_node_ids:
                    root_node_ids&=tmp_root_node_ids
                else:
                    root_node_ids=tmp_root_node_ids
            root_node_id=max(root_node_ids)
            code_layout_edges[0].extend(line_start_root_node_ids)
            code_layout_edges[1].extend([root_node_id]*len(line_start_root_node_ids))

        #将所有code_node_id连接到最近的 line_start_node_id
        for code_node_id,code_token_pos in zip(code_node_ids,code_token_poses):
            if line_start_node_ids[code_token_pos[0]] is not None:
                code_layout_edges[0].append(code_node_id)
                code_layout_edges[1].append(line_start_node_ids[code_token_pos[0]])
            else:
                child_node_id=code_node_id
                while child_node_id in ast_edges[0, :]:
                    father_node_id = ast_edges[1, :][ast_edges[0, :].tolist().index(child_node_id)]
                    if father_node_id in line_start_node_ids:
                        code_layout_edges[0].append(code_node_id) #这里不是child_node_id！
                        code_layout_edges[1].append(father_node_id)
                        break  # 找到即停止
                    child_node_id = father_node_id
        
        code_layout_edges=np.array(code_layout_edges)
        #叶子结点上加父节点
        # print(353 in code_node_ids)
        for code_node_id in code_node_ids:
            # if code_node_id==353:
            #     print(ast_edges[1,:][ast_edges[0,:].tolist().index(code_node_id)])
            #     print(code_layout_edges[1,:][code_layout_edges[0,:].tolist().index(code_node_id)])
            if self.ast_node_poses[code_node_id][-1]<0: #插入父节点
                # print(code_node_id)
                # print(code_layout_edges[0,:].tolist())
                del_layout_col_id=code_layout_edges[0,:].tolist().index(code_node_id)
                father_node_in_ast_id=ast_edges[1,:][ast_edges[0,:].tolist().index(code_node_id)]
                # if self.ast_nodes[father_node_in_ast_id]!='ERROR':
                code_layout_edges=np.concatenate([code_layout_edges,[[code_node_id],[father_node_in_ast_id]]],axis=-1)
                if father_node_in_ast_id not in code_layout_edges[0,:]: #如果在ast中的父节点不在layout子节点中 or code_layout_edges[1,:][code_layout_edges[0,:].tolist().index(father_node_in_ast_id)]!=father_node_in_layout_id
                    father_node_in_layout_id=code_layout_edges[1,:][code_layout_edges[0,:].tolist().index(code_node_id)]
                    # if  father_node_in_ast_id!=father_node_in_layout_id:
                    code_layout_edges=np.concatenate([code_layout_edges,[[father_node_in_ast_id],[father_node_in_layout_id]]],axis=-1)
                code_layout_edges=np.delete(code_layout_edges,del_layout_col_id,axis=-1)

        return code_layout_edges

    @property
    def code_layout_sibling_edges(self):
        code_layout_sibling_edges=deepcopy(self.ast_sibling_edges)
        ex_node_ids=sorted(set(np.unique(self.ast_edges))-set(np.unique(self.code_layout_edges)))
        # print(len(ex_node_ids))
        for ex_node_id in ex_node_ids: 
            if ex_node_id in self.ast_sibling_edges[0,:]:   #找到最左侧的合适子孙节点并替换
                right_child_id=max(self.ast_edges[0,:][np.argwhere(self.ast_edges[1,:]==ex_node_id)].reshape((-1,)))
                while right_child_id in ex_node_ids:
                    right_child_id=max(self.ast_edges[0,:][np.argwhere(self.ast_edges[1,:]==right_child_id)].reshape((-1,)))
                code_layout_sibling_edges[0,code_layout_sibling_edges[0,:].tolist().index(ex_node_id)]=right_child_id
            if ex_node_id in self.ast_sibling_edges[1,:]:   #找到最右侧的合适子孙节点并替换
                left_child_id=min(self.ast_edges[0,:][np.argwhere(self.ast_edges[1,:]==ex_node_id)].reshape((-1,)))
                while left_child_id in ex_node_ids:
                    left_child_id=min(self.ast_edges[0,:][np.argwhere(self.ast_edges[1,:]==left_child_id)].reshape((-1,)))
                code_layout_sibling_edges[1,code_layout_sibling_edges[1,:].tolist().index(ex_node_id)]=left_child_id
        return code_layout_sibling_edges


    @property
    def DFG_edges(self):
        ast_node_point2node_id_and_code=dict()
        # cur_ast_node_pos=(0,0,0)
        for node_id in self._pre_DFS_ids(self._intact_ast_edges):   #建DFG的时候必须为全树
            if node_id not in self._intact_ast_edges[1, :]:
                if self._intact_ast_node_poses[node_id][-1]>=0: #如果是功能节点
                    ast_node_point = self._intact_ast_node_points[node_id]
                    ast_node_point2node_id_and_code[ast_node_point]=[node_id,self._intact_ast_nodes[node_id]]
                else:   #如果是属性节点
                    father_id=self._intact_ast_edges[1,:][self._intact_ast_edges[0, :].tolist().index(node_id)]
                    ast_node_point=self._intact_ast_node_points[father_id]
                    if ast_node_point not in ast_node_point2node_id_and_code.keys():
                        ast_node_point2node_id_and_code[ast_node_point]=[father_id,'']
                        # cur_ast_node_pos=self._intact_ast_node_poses[node_id]
                    assert ast_node_point2node_id_and_code[ast_node_point][0]==father_id
                    ast_node_point2node_id_and_code[ast_node_point][1]+=''+self._intact_ast_nodes[node_id]
        DFG,DFG_edges=[],[[],[]]
        if self.language=='python':
            DFG, _ = DFG_python(root_node=self._tree.root_node, point2code=ast_node_point2node_id_and_code, states={})
        if self.language=='java':
            DFG, _ = DFG_java(root_node=self._tree.root_node, point2code=ast_node_point2node_id_and_code, states={})
        #DFG is the structure like this ('b', 11, 'comesFrom', [], []), ('x', 25, 'comesFrom', ['x'], [43, 49]), ('x', 26, 'computedFrom', ['0'], [28])
        if not self.ast_intact: #如果是精简树，建立全树节点id到简树节点id的映射，去掉冗余节点
            intact_ast_node_id2ast_concise_node_id=dict(zip([node_id for node_id,tags in
                                                             zip(list(range(len(self._intact_ast_nodes))),
                                                                 self._intact_redundant_ast_node_tags) if not tags],
                                                            list(range(len(self._concise_ast_nodes)))))
        for edge_obj in DFG:
            for start_node_id in edge_obj[4]:
                if self.ast_intact:
                    DFG_edges[0].append(start_node_id)
                    DFG_edges[1].append(edge_obj[1])
                elif start_node_id in intact_ast_node_id2ast_concise_node_id.keys() and \
                    edge_obj[1] in intact_ast_node_id2ast_concise_node_id.keys():   #如果是简树
                    DFG_edges[0].append(intact_ast_node_id2ast_concise_node_id[start_node_id])
                    DFG_edges[1].append(intact_ast_node_id2ast_concise_node_id[edge_obj[1]])

        return np.array(DFG_edges)