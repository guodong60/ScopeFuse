# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

# from tree_sitter import Language, Parser
# from .utils import (remove_comments_and_docstrings,
#                    tree_to_token_index,
#                    index_to_code_token,
#                    tree_to_variable_index)

# import re
# from io import StringIO
# import tokenize


def tree_to_token_index(root_node):
    if (len(root_node.children) == 0 or root_node.type == 'string') and root_node.type != 'comment':
        return [(root_node.start_point, root_node.end_point)]
    else:
        code_tokens = []
        for child in root_node.children:
            code_tokens += tree_to_token_index(child)
        return code_tokens


def tree_to_variable_index(root_node, point2code):
    if (len(root_node.children) == 0 or root_node.type == 'string') and root_node.type != 'comment':
        index = (root_node.start_point, root_node.end_point)
        _, code = point2code[index]
        if root_node.type != code:
            return [(root_node.start_point, root_node.end_point)]
        else:
            return []
    else:
        code_tokens = []
        for child in root_node.children:
            code_tokens += tree_to_variable_index(child, point2code)
        return code_tokens


def index_to_code_token(index, code):
    start_point = index[0]
    end_point = index[1]
    if start_point[0] == end_point[0]:
        s = code[start_point[0]][start_point[1]:end_point[1]]
    else:
        s = ""
        s += code[start_point[0]][start_point[1]:]
        for i in range(start_point[0] + 1, end_point[0]):
            s += code[i]
        s += code[end_point[0]][:end_point[1]]
    return s


def DFG_python(root_node,point2code,states):
    assignment=['assignment','augmented_assignment','for_in_clause']
    if_statement=['if_statement']
    for_statement=['for_statement']
    while_statement=['while_statement']
    do_first_statement=['for_in_clause'] 
    def_statement=['default_parameter']
    states=states.copy() 
    if (len(root_node.children)==0 or root_node.type=='string') and root_node.type!='comment':        
        idx,code=point2code[(root_node.start_point,root_node.end_point)]
        if root_node.type==code:
            return [],states
        elif code in states:
            return [(code,idx,'comesFrom',[code],states[code].copy())],states
        else:
            if root_node.type=='identifier':
                states[code]=[idx]
            return [(code,idx,'comesFrom',[],[])],states
    elif root_node.type in def_statement:
        name=root_node.child_by_field_name('name')
        value=root_node.child_by_field_name('value')
        DFG=[]
        if value is None:
            indexs=tree_to_variable_index(name,point2code)
            for index in indexs:
                idx,code=point2code[index]
                DFG.append((code,idx,'comesFrom',[],[]))
                states[code]=[idx]
            return sorted(DFG,key=lambda x:x[1]),states
        else:
            name_indexs=tree_to_variable_index(name,point2code)
            value_indexs=tree_to_variable_index(value,point2code)
            temp,states=DFG_python(value,point2code,states)
            DFG+=temp
            for index1 in name_indexs:
                idx1,code1=point2code[index1]
                for index2 in value_indexs:
                    idx2,code2=point2code[index2]
                    DFG.append((code1,idx1,'comesFrom',[code2],[idx2]))
                states[code1]=[idx1]
            return sorted(DFG,key=lambda x:x[1]),states
    elif root_node.type in assignment:
        if root_node.type=='for_in_clause':
            right_nodes=[root_node.children[-1]]
            left_nodes=[root_node.child_by_field_name('left')]
        else:
            if root_node.child_by_field_name('right') is None:
                return [],states
            left_nodes=[x for x in root_node.child_by_field_name('left').children if x.type!=',']
            right_nodes=[x for x in root_node.child_by_field_name('right').children if x.type!=',']
            if len(right_nodes)!=len(left_nodes):
                left_nodes=[root_node.child_by_field_name('left')]
                right_nodes=[root_node.child_by_field_name('right')]
            if len(left_nodes)==0:
                left_nodes=[root_node.child_by_field_name('left')]
            if len(right_nodes)==0:
                right_nodes=[root_node.child_by_field_name('right')]
        DFG=[]
        for node in right_nodes:
            temp,states=DFG_python(node,point2code,states)
            DFG+=temp

        for left_node,right_node in zip(left_nodes,right_nodes):
            left_tokens_index=tree_to_variable_index(left_node,point2code)
            right_tokens_index=tree_to_variable_index(right_node,point2code)
            temp=[]
            for token1_index in left_tokens_index:
                idx1,code1=point2code[token1_index]
                temp.append((code1,idx1,'computedFrom',[point2code[x][1] for x in right_tokens_index],
                             [point2code[x][0] for x in right_tokens_index]))
                states[code1]=[idx1]
            DFG+=temp
        return sorted(DFG,key=lambda x:x[1]),states
    elif root_node.type in if_statement:
        DFG=[]
        current_states=states.copy()
        others_states=[]
        tag=False
        if 'else' in root_node.type:
            tag=True
        for child in root_node.children:
            if 'else' in child.type:
                tag=True
            if child.type not in ['elif_clause','else_clause']:
                temp,current_states=DFG_python(child,point2code,current_states)
                DFG+=temp
            else:
                temp,new_states=DFG_python(child,point2code,states)
                DFG+=temp
                others_states.append(new_states)
        others_states.append(current_states)
        if tag is False:
            others_states.append(states)
        new_states={}
        for dic in others_states:
            for key in dic:
                if key not in new_states:
                    new_states[key]=dic[key].copy()
                else:
                    new_states[key]+=dic[key]
        for key in new_states:
            new_states[key]=sorted(list(set(new_states[key])))
        return sorted(DFG,key=lambda x:x[1]),new_states
    elif root_node.type in for_statement:
        DFG=[]
        for i in range(2):
            right_nodes=[x for x in root_node.child_by_field_name('right').children if x.type!=',']
            left_nodes=[x for x in root_node.child_by_field_name('left').children if x.type!=',']
            if len(right_nodes)!=len(left_nodes):
                left_nodes=[root_node.child_by_field_name('left')]
                right_nodes=[root_node.child_by_field_name('right')]
            if len(left_nodes)==0:
                left_nodes=[root_node.child_by_field_name('left')]
            if len(right_nodes)==0:
                right_nodes=[root_node.child_by_field_name('right')]
            for node in right_nodes:
                temp,states=DFG_python(node,point2code,states)
                DFG+=temp
            for left_node,right_node in zip(left_nodes,right_nodes):
                left_tokens_index=tree_to_variable_index(left_node,point2code)
                right_tokens_index=tree_to_variable_index(right_node,point2code)
                temp=[]
                for token1_index in left_tokens_index:
                    idx1,code1=point2code[token1_index]
                    temp.append((code1,idx1,'computedFrom',[point2code[x][1] for x in right_tokens_index],
                                 [point2code[x][0] for x in right_tokens_index]))
                    states[code1]=[idx1]
                DFG+=temp
            if  root_node.children[-1].type=="block":
                temp,states=DFG_python(root_node.children[-1],point2code,states)
                DFG+=temp
        dic={}
        for x in DFG:
            if (x[0],x[1],x[2]) not in dic:
                dic[(x[0],x[1],x[2])]=[x[3],x[4]]
            else:
                dic[(x[0],x[1],x[2])][0]=list(set(dic[(x[0],x[1],x[2])][0]+x[3]))
                dic[(x[0],x[1],x[2])][1]=sorted(list(set(dic[(x[0],x[1],x[2])][1]+x[4])))
        DFG=[(x[0],x[1],x[2],y[0],y[1]) for x,y in sorted(dic.items(),key=lambda t:t[0][1])]
        return sorted(DFG,key=lambda x:x[1]),states
    elif root_node.type in while_statement:
        DFG=[]
        for i in range(2):
            for child in root_node.children:
                temp,states=DFG_python(child,point2code,states)
                DFG+=temp
        dic={}
        for x in DFG:
            if (x[0],x[1],x[2]) not in dic:
                dic[(x[0],x[1],x[2])]=[x[3],x[4]]
            else:
                dic[(x[0],x[1],x[2])][0]=list(set(dic[(x[0],x[1],x[2])][0]+x[3]))
                dic[(x[0],x[1],x[2])][1]=sorted(list(set(dic[(x[0],x[1],x[2])][1]+x[4])))
        DFG=[(x[0],x[1],x[2],y[0],y[1]) for x,y in sorted(dic.items(),key=lambda t:t[0][1])]
        return sorted(DFG,key=lambda x:x[1]),states
    else:
        DFG=[]
        for child in root_node.children:
            if child.type in do_first_statement:
                temp,states=DFG_python(child,point2code,states)
                DFG+=temp
        for child in root_node.children:
            if child.type not in do_first_statement:
                temp,states=DFG_python(child,point2code,states)
                DFG+=temp

        return sorted(DFG,key=lambda x:x[1]),states

def DFG_java(root_node,point2code,states):
    assignment=['assignment_expression']
    def_statement=['variable_declarator']
    increment_statement=['update_expression']
    if_statement=['if_statement','else']
    for_statement=['for_statement']
    enhanced_for_statement=['enhanced_for_statement']
    while_statement=['while_statement']
    do_first_statement=[]
    states=states.copy()
    if (len(root_node.children)==0 or root_node.type=='string') and root_node.type!='line_comment': #java代码中是line_comment
        idx,code=point2code[(root_node.start_point,root_node.end_point)]
        if root_node.type==code:
            return [],states
        elif code in states:
            return [(code,idx,'comesFrom',[code],states[code].copy())],states
        else:
            if root_node.type=='identifier':
                states[code]=[idx]
            return [(code,idx,'comesFrom',[],[])],states
    elif root_node.type in def_statement:
        name=root_node.child_by_field_name('name')
        value=root_node.child_by_field_name('value')
        DFG=[]
        if value is None:
            indexs=tree_to_variable_index(name,point2code)
            for index in indexs:
                idx,code=point2code[index]
                DFG.append((code,idx,'comesFrom',[],[]))
                states[code]=[idx]
            return sorted(DFG,key=lambda x:x[1]),states
        else:
            name_indexs=tree_to_variable_index(name,point2code)
            value_indexs=tree_to_variable_index(value,point2code)
            temp,states=DFG_java(value,point2code,states)
            DFG+=temp
            for index1 in name_indexs:
                idx1,code1=point2code[index1]
                for index2 in value_indexs:
                    idx2,code2=point2code[index2]
                    DFG.append((code1,idx1,'comesFrom',[code2],[idx2]))
                states[code1]=[idx1]
            return sorted(DFG,key=lambda x:x[1]),states
    elif root_node.type in assignment:
        left_nodes=root_node.child_by_field_name('left')
        right_nodes=root_node.child_by_field_name('right')
        DFG=[]
        temp,states=DFG_java(right_nodes,point2code,states)
        DFG+=temp
        name_indexs=tree_to_variable_index(left_nodes,point2code)
        value_indexs=tree_to_variable_index(right_nodes,point2code)
        for index1 in name_indexs:
            idx1,code1=point2code[index1]
            for index2 in value_indexs:
                idx2,code2=point2code[index2]
                DFG.append((code1,idx1,'computedFrom',[code2],[idx2]))
            states[code1]=[idx1]
        return sorted(DFG,key=lambda x:x[1]),states
    elif root_node.type in increment_statement:
        DFG=[]
        indexs=tree_to_variable_index(root_node,point2code)
        for index1 in indexs:
            idx1,code1=point2code[index1]
            for index2 in indexs:
                idx2,code2=point2code[index2]
                DFG.append((code1,idx1,'computedFrom',[code2],[idx2]))
            states[code1]=[idx1]
        return sorted(DFG,key=lambda x:x[1]),states
    elif root_node.type in if_statement:
        DFG=[]
        current_states=states.copy()
        others_states=[]
        flag=False
        tag=False
        if 'else' in root_node.type:
            tag=True
        for child in root_node.children:
            if 'else' in child.type:
                tag=True
            if child.type not in if_statement and flag is False:
                temp,current_states=DFG_java(child,point2code,current_states)
                DFG+=temp
            else:
                flag=True
                temp,new_states=DFG_java(child,point2code,states)
                DFG+=temp
                others_states.append(new_states)
        others_states.append(current_states)
        if tag is False:
            others_states.append(states)
        new_states={}
        for dic in others_states:
            for key in dic:
                if key not in new_states:
                    new_states[key]=dic[key].copy()
                else:
                    new_states[key]+=dic[key]
        for key in new_states:
            new_states[key]=sorted(list(set(new_states[key])))
        return sorted(DFG,key=lambda x:x[1]),new_states
    elif root_node.type in for_statement:
        DFG=[]
        for child in root_node.children:
            temp,states=DFG_java(child,point2code,states)
            DFG+=temp
        flag=False
        for child in root_node.children:
            if flag:
                temp,states=DFG_java(child,point2code,states)
                DFG+=temp
            elif child.type=="local_variable_declaration":
                flag=True
        dic={}
        for x in DFG:
            if (x[0],x[1],x[2]) not in dic:
                dic[(x[0],x[1],x[2])]=[x[3],x[4]]
            else:
                dic[(x[0],x[1],x[2])][0]=list(set(dic[(x[0],x[1],x[2])][0]+x[3]))
                dic[(x[0],x[1],x[2])][1]=sorted(list(set(dic[(x[0],x[1],x[2])][1]+x[4])))
        DFG=[(x[0],x[1],x[2],y[0],y[1]) for x,y in sorted(dic.items(),key=lambda t:t[0][1])]
        return sorted(DFG,key=lambda x:x[1]),states
    elif root_node.type in enhanced_for_statement:
        name=root_node.child_by_field_name('name')
        value=root_node.child_by_field_name('value')
        body=root_node.child_by_field_name('body')
        DFG=[]
        for i in range(2):
            temp,states=DFG_java(value,point2code,states)
            DFG+=temp
            name_indexs=tree_to_variable_index(name,point2code)
            value_indexs=tree_to_variable_index(value,point2code)
            for index1 in name_indexs:
                idx1,code1=point2code[index1]
                for index2 in value_indexs:
                    idx2,code2=point2code[index2]
                    DFG.append((code1,idx1,'computedFrom',[code2],[idx2]))
                states[code1]=[idx1]
            temp,states=DFG_java(body,point2code,states)
            DFG+=temp
        dic={}
        for x in DFG:
            if (x[0],x[1],x[2]) not in dic:
                dic[(x[0],x[1],x[2])]=[x[3],x[4]]
            else:
                dic[(x[0],x[1],x[2])][0]=list(set(dic[(x[0],x[1],x[2])][0]+x[3]))
                dic[(x[0],x[1],x[2])][1]=sorted(list(set(dic[(x[0],x[1],x[2])][1]+x[4])))
        DFG=[(x[0],x[1],x[2],y[0],y[1]) for x,y in sorted(dic.items(),key=lambda t:t[0][1])]
        return sorted(DFG,key=lambda x:x[1]),states
    elif root_node.type in while_statement:
        DFG=[]
        for i in range(2):
            for child in root_node.children:
                temp,states=DFG_java(child,point2code,states)
                DFG+=temp
        dic={}
        for x in DFG:
            if (x[0],x[1],x[2]) not in dic:
                dic[(x[0],x[1],x[2])]=[x[3],x[4]]
            else:
                dic[(x[0],x[1],x[2])][0]=list(set(dic[(x[0],x[1],x[2])][0]+x[3]))
                dic[(x[0],x[1],x[2])][1]=sorted(list(set(dic[(x[0],x[1],x[2])][1]+x[4])))
        DFG=[(x[0],x[1],x[2],y[0],y[1]) for x,y in sorted(dic.items(),key=lambda t:t[0][1])]
        return sorted(DFG,key=lambda x:x[1]),states
    else:
        DFG=[]
        for child in root_node.children:
            if child.type in do_first_statement:
                temp,states=DFG_java(child,point2code,states)
                DFG+=temp
        for child in root_node.children:
            if child.type not in do_first_statement:
                temp,states=DFG_java(child,point2code,states)
                DFG+=temp

        return sorted(DFG,key=lambda x:x[1]),states


def DFG_csharp(root_node,point2code,states):
    assignment=['assignment_expression']
    def_statement=['variable_declarator']
    increment_statement=['postfix_unary_expression']
    if_statement=['if_statement','else']
    for_statement=['for_statement']
    enhanced_for_statement=['for_each_statement']
    while_statement=['while_statement']
    do_first_statement=[]
    states=states.copy()
    if (len(root_node.children)==0 or root_node.type=='string') and root_node.type!='comment':
        idx,code=point2code[(root_node.start_point,root_node.end_point)]
        if root_node.type==code:
            return [],states
        elif code in states:
            return [(code,idx,'comesFrom',[code],states[code].copy())],states
        else:
            if root_node.type=='identifier':
                states[code]=[idx]
            return [(code,idx,'comesFrom',[],[])],states
    elif root_node.type in def_statement:
        if len(root_node.children)==2:
            name=root_node.children[0]
            value=root_node.children[1]
        else:
            name=root_node.children[0]
            value=None
        DFG=[]
        if value is None:
            indexs=tree_to_variable_index(name,point2code)
            for index in indexs:
                idx,code=point2code[index]
                DFG.append((code,idx,'comesFrom',[],[]))
                states[code]=[idx]
            return sorted(DFG,key=lambda x:x[1]),states
        else:
            name_indexs=tree_to_variable_index(name,point2code)
            value_indexs=tree_to_variable_index(value,point2code)
            temp,states=DFG_csharp(value,point2code,states)
            DFG+=temp
            for index1 in name_indexs:
                idx1,code1=point2code[index1]
                for index2 in value_indexs:
                    idx2,code2=point2code[index2]
                    DFG.append((code1,idx1,'comesFrom',[code2],[idx2]))
                states[code1]=[idx1]
            return sorted(DFG,key=lambda x:x[1]),states
    elif root_node.type in assignment:
        left_nodes=root_node.child_by_field_name('left')
        right_nodes=root_node.child_by_field_name('right')
        DFG=[]
        temp,states=DFG_csharp(right_nodes,point2code,states)
        DFG+=temp
        name_indexs=tree_to_variable_index(left_nodes,point2code)
        value_indexs=tree_to_variable_index(right_nodes,point2code)
        for index1 in name_indexs:
            idx1,code1=point2code[index1]
            for index2 in value_indexs:
                idx2,code2=point2code[index2]
                DFG.append((code1,idx1,'computedFrom',[code2],[idx2]))
            states[code1]=[idx1]
        return sorted(DFG,key=lambda x:x[1]),states
    elif root_node.type in increment_statement:
        DFG=[]
        indexs=tree_to_variable_index(root_node,point2code)
        for index1 in indexs:
            idx1,code1=point2code[index1]
            for index2 in indexs:
                idx2,code2=point2code[index2]
                DFG.append((code1,idx1,'computedFrom',[code2],[idx2]))
            states[code1]=[idx1]
        return sorted(DFG,key=lambda x:x[1]),states
    elif root_node.type in if_statement:
        DFG=[]
        current_states=states.copy()
        others_states=[]
        flag=False
        tag=False
        if 'else' in root_node.type:
            tag=True
        for child in root_node.children:
            if 'else' in child.type:
                tag=True
            if child.type not in if_statement and flag is False:
                temp,current_states=DFG_csharp(child,point2code,current_states)
                DFG+=temp
            else:
                flag=True
                temp,new_states=DFG_csharp(child,point2code,states)
                DFG+=temp
                others_states.append(new_states)
        others_states.append(current_states)
        if tag is False:
            others_states.append(states)
        new_states={}
        for dic in others_states:
            for key in dic:
                if key not in new_states:
                    new_states[key]=dic[key].copy()
                else:
                    new_states[key]+=dic[key]
        for key in new_states:
            new_states[key]=sorted(list(set(new_states[key])))
        return sorted(DFG,key=lambda x:x[1]),new_states
    elif root_node.type in for_statement:
        DFG=[]
        for child in root_node.children:
            temp,states=DFG_csharp(child,point2code,states)
            DFG+=temp
        flag=False
        for child in root_node.children:
            if flag:
                temp,states=DFG_csharp(child,point2code,states)
                DFG+=temp
            elif child.type=="local_variable_declaration":
                flag=True
        dic={}
        for x in DFG:
            if (x[0],x[1],x[2]) not in dic:
                dic[(x[0],x[1],x[2])]=[x[3],x[4]]
            else:
                dic[(x[0],x[1],x[2])][0]=list(set(dic[(x[0],x[1],x[2])][0]+x[3]))
                dic[(x[0],x[1],x[2])][1]=sorted(list(set(dic[(x[0],x[1],x[2])][1]+x[4])))
        DFG=[(x[0],x[1],x[2],y[0],y[1]) for x,y in sorted(dic.items(),key=lambda t:t[0][1])]
        return sorted(DFG,key=lambda x:x[1]),states
    elif root_node.type in enhanced_for_statement:
        name=root_node.child_by_field_name('left')
        value=root_node.child_by_field_name('right')
        body=root_node.child_by_field_name('body')
        DFG=[]
        for i in range(2):
            temp,states=DFG_csharp(value,point2code,states)
            DFG+=temp
            name_indexs=tree_to_variable_index(name,point2code)
            value_indexs=tree_to_variable_index(value,point2code)
            for index1 in name_indexs:
                idx1,code1=point2code[index1]
                for index2 in value_indexs:
                    idx2,code2=point2code[index2]
                    DFG.append((code1,idx1,'computedFrom',[code2],[idx2]))
                states[code1]=[idx1]
            temp,states=DFG_csharp(body,point2code,states)
            DFG+=temp
        dic={}
        for x in DFG:
            if (x[0],x[1],x[2]) not in dic:
                dic[(x[0],x[1],x[2])]=[x[3],x[4]]
            else:
                dic[(x[0],x[1],x[2])][0]=list(set(dic[(x[0],x[1],x[2])][0]+x[3]))
                dic[(x[0],x[1],x[2])][1]=sorted(list(set(dic[(x[0],x[1],x[2])][1]+x[4])))
        DFG=[(x[0],x[1],x[2],y[0],y[1]) for x,y in sorted(dic.items(),key=lambda t:t[0][1])]
        return sorted(DFG,key=lambda x:x[1]),states
    elif root_node.type in while_statement:
        DFG=[]
        for i in range(2):
            for child in root_node.children:
                temp,states=DFG_csharp(child,point2code,states)
                DFG+=temp
        dic={}
        for x in DFG:
            if (x[0],x[1],x[2]) not in dic:
                dic[(x[0],x[1],x[2])]=[x[3],x[4]]
            else:
                dic[(x[0],x[1],x[2])][0]=list(set(dic[(x[0],x[1],x[2])][0]+x[3]))
                dic[(x[0],x[1],x[2])][1]=sorted(list(set(dic[(x[0],x[1],x[2])][1]+x[4])))
        DFG=[(x[0],x[1],x[2],y[0],y[1]) for x,y in sorted(dic.items(),key=lambda t:t[0][1])]
        return sorted(DFG,key=lambda x:x[1]),states
    else:
        DFG=[]
        for child in root_node.children:
            if child.type in do_first_statement:
                temp,states=DFG_csharp(child,point2code,states)
                DFG+=temp
        for child in root_node.children:
            if child.type not in do_first_statement:
                temp,states=DFG_csharp(child,point2code,states)
                DFG+=temp

        return sorted(DFG,key=lambda x:x[1]),states

def DFG_ruby(root_node,point2code,states):
    assignment=['assignment','operator_assignment']
    if_statement=['if','elsif','else','unless','when']
    for_statement=['for']
    while_statement=['while_modifier','until']
    do_first_statement=[]
    def_statement=['keyword_parameter']
    if (len(root_node.children)==0 or root_node.type=='string') and root_node.type!='comment':
        states=states.copy()
        idx,code=point2code[(root_node.start_point,root_node.end_point)]
        if root_node.type==code:
            return [],states
        elif code in states:
            return [(code,idx,'comesFrom',[code],states[code].copy())],states
        else:
            if root_node.type=='identifier':
                states[code]=[idx]
            return [(code,idx,'comesFrom',[],[])],states
    elif root_node.type in def_statement:
        name=root_node.child_by_field_name('name')
        value=root_node.child_by_field_name('value')
        DFG=[]
        if value is None:
            indexs=tree_to_variable_index(name,point2code)
            for index in indexs:
                idx,code=point2code[index]
                DFG.append((code,idx,'comesFrom',[],[]))
                states[code]=[idx]
            return sorted(DFG,key=lambda x:x[1]),states
        else:
            name_indexs=tree_to_variable_index(name,point2code)
            value_indexs=tree_to_variable_index(value,point2code)
            temp,states=DFG_ruby(value,point2code,states)
            DFG+=temp
            for index1 in name_indexs:
                idx1,code1=point2code[index1]
                for index2 in value_indexs:
                    idx2,code2=point2code[index2]
                    DFG.append((code1,idx1,'comesFrom',[code2],[idx2]))
                states[code1]=[idx1]
            return sorted(DFG,key=lambda x:x[1]),states
    elif root_node.type in assignment:
        left_nodes=[x for x in root_node.child_by_field_name('left').children if x.type!=',']
        right_nodes=[x for x in root_node.child_by_field_name('right').children if x.type!=',']
        if len(right_nodes)!=len(left_nodes):
            left_nodes=[root_node.child_by_field_name('left')]
            right_nodes=[root_node.child_by_field_name('right')]
        if len(left_nodes)==0:
            left_nodes=[root_node.child_by_field_name('left')]
        if len(right_nodes)==0:
            right_nodes=[root_node.child_by_field_name('right')]
        if root_node.type=="operator_assignment":
            left_nodes=[root_node.children[0]]
            right_nodes=[root_node.children[-1]]

        DFG=[]
        for node in right_nodes:
            temp,states=DFG_ruby(node,point2code,states)
            DFG+=temp

        for left_node,right_node in zip(left_nodes,right_nodes):
            left_tokens_index=tree_to_variable_index(left_node,point2code)
            right_tokens_index=tree_to_variable_index(right_node,point2code)
            temp=[]
            for token1_index in left_tokens_index:
                idx1,code1=point2code[token1_index]
                temp.append((code1,idx1,'computedFrom',[point2code[x][1] for x in right_tokens_index],
                             [point2code[x][0] for x in right_tokens_index]))
                states[code1]=[idx1]
            DFG+=temp
        return sorted(DFG,key=lambda x:x[1]),states
    elif root_node.type in if_statement:
        DFG=[]
        current_states=states.copy()
        others_states=[]
        tag=False
        if 'else' in root_node.type:
            tag=True
        for child in root_node.children:
            if 'else' in child.type:
                tag=True
            if child.type not in if_statement:
                temp,current_states=DFG_ruby(child,point2code,current_states)
                DFG+=temp
            else:
                temp,new_states=DFG_ruby(child,point2code,states)
                DFG+=temp
                others_states.append(new_states)
        others_states.append(current_states)
        if tag is False:
            others_states.append(states)
        new_states={}
        for dic in others_states:
            for key in dic:
                if key not in new_states:
                    new_states[key]=dic[key].copy()
                else:
                    new_states[key]+=dic[key]
        for key in new_states:
            new_states[key]=sorted(list(set(new_states[key])))
        return sorted(DFG,key=lambda x:x[1]),new_states
    elif root_node.type in for_statement:
        DFG=[]
        for i in range(2):
            left_nodes=[root_node.child_by_field_name('pattern')]
            right_nodes=[root_node.child_by_field_name('value')]
            assert len(right_nodes)==len(left_nodes)
            for node in right_nodes:
                temp,states=DFG_ruby(node,point2code,states)
                DFG+=temp
            for left_node,right_node in zip(left_nodes,right_nodes):
                left_tokens_index=tree_to_variable_index(left_node,point2code)
                right_tokens_index=tree_to_variable_index(right_node,point2code)
                temp=[]
                for token1_index in left_tokens_index:
                    idx1,code1=point2code[token1_index]
                    temp.append((code1,idx1,'computedFrom',[point2code[x][1] for x in right_tokens_index],
                                 [point2code[x][0] for x in right_tokens_index]))
                    states[code1]=[idx1]
                DFG+=temp
            temp,states=DFG_ruby(root_node.child_by_field_name('body'),point2code,states)
            DFG+=temp
        dic={}
        for x in DFG:
            if (x[0],x[1],x[2]) not in dic:
                dic[(x[0],x[1],x[2])]=[x[3],x[4]]
            else:
                dic[(x[0],x[1],x[2])][0]=list(set(dic[(x[0],x[1],x[2])][0]+x[3]))
                dic[(x[0],x[1],x[2])][1]=sorted(list(set(dic[(x[0],x[1],x[2])][1]+x[4])))
        DFG=[(x[0],x[1],x[2],y[0],y[1]) for x,y in sorted(dic.items(),key=lambda t:t[0][1])]
        return sorted(DFG,key=lambda x:x[1]),states
    elif root_node.type in while_statement:
        DFG=[]
        for i in range(2):
            for child in root_node.children:
                temp,states=DFG_ruby(child,point2code,states)
                DFG+=temp
        dic={}
        for x in DFG:
            if (x[0],x[1],x[2]) not in dic:
                dic[(x[0],x[1],x[2])]=[x[3],x[4]]
            else:
                dic[(x[0],x[1],x[2])][0]=list(set(dic[(x[0],x[1],x[2])][0]+x[3]))
                dic[(x[0],x[1],x[2])][1]=sorted(list(set(dic[(x[0],x[1],x[2])][1]+x[4])))
        DFG=[(x[0],x[1],x[2],y[0],y[1]) for x,y in sorted(dic.items(),key=lambda t:t[0][1])]
        return sorted(DFG,key=lambda x:x[1]),states
    else:
        DFG=[]
        for child in root_node.children:
            if child.type in do_first_statement:
                temp,states=DFG_ruby(child,point2code,states)
                DFG+=temp
        for child in root_node.children:
            if child.type not in do_first_statement:
                temp,states=DFG_ruby(child,point2code,states)
                DFG+=temp

        return sorted(DFG,key=lambda x:x[1]),states

def DFG_go(root_node,point2code,states):
    assignment=['assignment_statement',]
    def_statement=['var_spec']
    increment_statement=['inc_statement']
    if_statement=['if_statement','else']
    for_statement=['for_statement']
    enhanced_for_statement=[]
    while_statement=[]
    do_first_statement=[]
    states=states.copy()
    if (len(root_node.children)==0 or root_node.type=='string') and root_node.type!='comment':
        idx,code=point2code[(root_node.start_point,root_node.end_point)]
        if root_node.type==code:
            return [],states
        elif code in states:
            return [(code,idx,'comesFrom',[code],states[code].copy())],states
        else:
            if root_node.type=='identifier':
                states[code]=[idx]
            return [(code,idx,'comesFrom',[],[])],states
    elif root_node.type in def_statement:
        name=root_node.child_by_field_name('name')
        value=root_node.child_by_field_name('value')
        DFG=[]
        if value is None:
            indexs=tree_to_variable_index(name,point2code)
            for index in indexs:
                idx,code=point2code[index]
                DFG.append((code,idx,'comesFrom',[],[]))
                states[code]=[idx]
            return sorted(DFG,key=lambda x:x[1]),states
        else:
            name_indexs=tree_to_variable_index(name,point2code)
            value_indexs=tree_to_variable_index(value,point2code)
            temp,states=DFG_go(value,point2code,states)
            DFG+=temp
            for index1 in name_indexs:
                idx1,code1=point2code[index1]
                for index2 in value_indexs:
                    idx2,code2=point2code[index2]
                    DFG.append((code1,idx1,'comesFrom',[code2],[idx2]))
                states[code1]=[idx1]
            return sorted(DFG,key=lambda x:x[1]),states
    elif root_node.type in assignment:
        left_nodes=root_node.child_by_field_name('left')
        right_nodes=root_node.child_by_field_name('right')
        DFG=[]
        temp,states=DFG_go(right_nodes,point2code,states)
        DFG+=temp
        name_indexs=tree_to_variable_index(left_nodes,point2code)
        value_indexs=tree_to_variable_index(right_nodes,point2code)
        for index1 in name_indexs:
            idx1,code1=point2code[index1]
            for index2 in value_indexs:
                idx2,code2=point2code[index2]
                DFG.append((code1,idx1,'computedFrom',[code2],[idx2]))
            states[code1]=[idx1]
        return sorted(DFG,key=lambda x:x[1]),states
    elif root_node.type in increment_statement:
        DFG=[]
        indexs=tree_to_variable_index(root_node,point2code)
        for index1 in indexs:
            idx1,code1=point2code[index1]
            for index2 in indexs:
                idx2,code2=point2code[index2]
                DFG.append((code1,idx1,'computedFrom',[code2],[idx2]))
            states[code1]=[idx1]
        return sorted(DFG,key=lambda x:x[1]),states
    elif root_node.type in if_statement:
        DFG=[]
        current_states=states.copy()
        others_states=[]
        flag=False
        tag=False
        if 'else' in root_node.type:
            tag=True
        for child in root_node.children:
            if 'else' in child.type:
                tag=True
            if child.type not in if_statement and flag is False:
                temp,current_states=DFG_go(child,point2code,current_states)
                DFG+=temp
            else:
                flag=True
                temp,new_states=DFG_go(child,point2code,states)
                DFG+=temp
                others_states.append(new_states)
        others_states.append(current_states)
        if tag is False:
            others_states.append(states)
        new_states={}
        for dic in others_states:
            for key in dic:
                if key not in new_states:
                    new_states[key]=dic[key].copy()
                else:
                    new_states[key]+=dic[key]
        for key in states:
            if key not in new_states:
                new_states[key]=states[key]
            else:
                new_states[key]+=states[key]
        for key in new_states:
            new_states[key]=sorted(list(set(new_states[key])))
        return sorted(DFG,key=lambda x:x[1]),new_states
    elif root_node.type in for_statement:
        DFG=[]
        for child in root_node.children:
            temp,states=DFG_go(child,point2code,states)
            DFG+=temp
        flag=False
        for child in root_node.children:
            if flag:
                temp,states=DFG_go(child,point2code,states)
                DFG+=temp
            elif child.type=="for_clause":
                if child.child_by_field_name('update') is not None:
                    temp,states=DFG_go(child.child_by_field_name('update'),point2code,states)
                    DFG+=temp
                flag=True
        dic={}
        for x in DFG:
            if (x[0],x[1],x[2]) not in dic:
                dic[(x[0],x[1],x[2])]=[x[3],x[4]]
            else:
                dic[(x[0],x[1],x[2])][0]=list(set(dic[(x[0],x[1],x[2])][0]+x[3]))
                dic[(x[0],x[1],x[2])][1]=sorted(list(set(dic[(x[0],x[1],x[2])][1]+x[4])))
        DFG=[(x[0],x[1],x[2],y[0],y[1]) for x,y in sorted(dic.items(),key=lambda t:t[0][1])]
        return sorted(DFG,key=lambda x:x[1]),states
    else:
        DFG=[]
        for child in root_node.children:
            if child.type in do_first_statement:
                temp,states=DFG_go(child,point2code,states)
                DFG+=temp
        for child in root_node.children:
            if child.type not in do_first_statement:
                temp,states=DFG_go(child,point2code,states)
                DFG+=temp

        return sorted(DFG,key=lambda x:x[1]),states

def DFG_php(root_node,point2code,states):
    assignment=['assignment_expression','augmented_assignment_expression']
    def_statement=['simple_parameter']
    increment_statement=['update_expression']
    if_statement=['if_statement','else_clause']
    for_statement=['for_statement']
    enhanced_for_statement=['foreach_statement']
    while_statement=['while_statement']
    do_first_statement=[]
    states=states.copy()
    if (len(root_node.children)==0 or root_node.type=='string') and root_node.type!='comment':
        idx,code=point2code[(root_node.start_point,root_node.end_point)]
        if root_node.type==code:
            return [],states
        elif code in states:
            return [(code,idx,'comesFrom',[code],states[code].copy())],states
        else:
            if root_node.type=='identifier':
                states[code]=[idx]
            return [(code,idx,'comesFrom',[],[])],states
    elif root_node.type in def_statement:
        name=root_node.child_by_field_name('name')
        value=root_node.child_by_field_name('default_value')
        DFG=[]
        if value is None:
            indexs=tree_to_variable_index(name,point2code)
            for index in indexs:
                idx,code=point2code[index]
                DFG.append((code,idx,'comesFrom',[],[]))
                states[code]=[idx]
            return sorted(DFG,key=lambda x:x[1]),states
        else:
            name_indexs=tree_to_variable_index(name,point2code)
            value_indexs=tree_to_variable_index(value,point2code)
            temp,states=DFG_php(value,point2code,states)
            DFG+=temp
            for index1 in name_indexs:
                idx1,code1=point2code[index1]
                for index2 in value_indexs:
                    idx2,code2=point2code[index2]
                    DFG.append((code1,idx1,'comesFrom',[code2],[idx2]))
                states[code1]=[idx1]
            return sorted(DFG,key=lambda x:x[1]),states
    elif root_node.type in assignment:
        left_nodes=root_node.child_by_field_name('left')
        right_nodes=root_node.child_by_field_name('right')
        DFG=[]
        temp,states=DFG_php(right_nodes,point2code,states)
        DFG+=temp
        name_indexs=tree_to_variable_index(left_nodes,point2code)
        value_indexs=tree_to_variable_index(right_nodes,point2code)
        for index1 in name_indexs:
            idx1,code1=point2code[index1]
            for index2 in value_indexs:
                idx2,code2=point2code[index2]
                DFG.append((code1,idx1,'computedFrom',[code2],[idx2]))
            states[code1]=[idx1]
        return sorted(DFG,key=lambda x:x[1]),states
    elif root_node.type in increment_statement:
        DFG=[]
        indexs=tree_to_variable_index(root_node,point2code)
        for index1 in indexs:
            idx1,code1=point2code[index1]
            for index2 in indexs:
                idx2,code2=point2code[index2]
                DFG.append((code1,idx1,'computedFrom',[code2],[idx2]))
            states[code1]=[idx1]
        return sorted(DFG,key=lambda x:x[1]),states
    elif root_node.type in if_statement:
        DFG=[]
        current_states=states.copy()
        others_states=[]
        flag=False
        tag=False
        if 'else' in root_node.type:
            tag=True
        for child in root_node.children:
            if 'else' in child.type:
                tag=True
            if child.type not in if_statement and flag is False:
                temp,current_states=DFG_php(child,point2code,current_states)
                DFG+=temp
            else:
                flag=True
                temp,new_states=DFG_php(child,point2code,states)
                DFG+=temp
                others_states.append(new_states)
        others_states.append(current_states)
        new_states={}
        for dic in others_states:
            for key in dic:
                if key not in new_states:
                    new_states[key]=dic[key].copy()
                else:
                    new_states[key]+=dic[key]
        for key in states:
            if key not in new_states:
                new_states[key]=states[key]
            else:
                new_states[key]+=states[key]
        for key in new_states:
            new_states[key]=sorted(list(set(new_states[key])))
        return sorted(DFG,key=lambda x:x[1]),new_states
    elif root_node.type in for_statement:
        DFG=[]
        for child in root_node.children:
            temp,states=DFG_php(child,point2code,states)
            DFG+=temp
        flag=False
        for child in root_node.children:
            if flag:
                temp,states=DFG_php(child,point2code,states)
                DFG+=temp
            elif child.type=="assignment_expression":
                flag=True
        dic={}
        for x in DFG:
            if (x[0],x[1],x[2]) not in dic:
                dic[(x[0],x[1],x[2])]=[x[3],x[4]]
            else:
                dic[(x[0],x[1],x[2])][0]=list(set(dic[(x[0],x[1],x[2])][0]+x[3]))
                dic[(x[0],x[1],x[2])][1]=sorted(list(set(dic[(x[0],x[1],x[2])][1]+x[4])))
        DFG=[(x[0],x[1],x[2],y[0],y[1]) for x,y in sorted(dic.items(),key=lambda t:t[0][1])]
        return sorted(DFG,key=lambda x:x[1]),states
    elif root_node.type in enhanced_for_statement:
        name=None
        value=None
        for child in root_node.children:
            if child.type=='variable_name' and value is None:
                value=child
            elif child.type=='variable_name' and name is None:
                name=child
                break
        body=root_node.child_by_field_name('body')
        DFG=[]
        for i in range(2):
            temp,states=DFG_php(value,point2code,states)
            DFG+=temp
            name_indexs=tree_to_variable_index(name,point2code)
            value_indexs=tree_to_variable_index(value,point2code)
            for index1 in name_indexs:
                idx1,code1=point2code[index1]
                for index2 in value_indexs:
                    idx2,code2=point2code[index2]
                    DFG.append((code1,idx1,'computedFrom',[code2],[idx2]))
                states[code1]=[idx1]
            temp,states=DFG_php(body,point2code,states)
            DFG+=temp
        dic={}
        for x in DFG:
            if (x[0],x[1],x[2]) not in dic:
                dic[(x[0],x[1],x[2])]=[x[3],x[4]]
            else:
                dic[(x[0],x[1],x[2])][0]=list(set(dic[(x[0],x[1],x[2])][0]+x[3]))
                dic[(x[0],x[1],x[2])][1]=sorted(list(set(dic[(x[0],x[1],x[2])][1]+x[4])))
        DFG=[(x[0],x[1],x[2],y[0],y[1]) for x,y in sorted(dic.items(),key=lambda t:t[0][1])]
        return sorted(DFG,key=lambda x:x[1]),states
    elif root_node.type in while_statement:
        DFG=[]
        for i in range(2):
            for child in root_node.children:
                temp,states=DFG_php(child,point2code,states)
                DFG+=temp
        dic={}
        for x in DFG:
            if (x[0],x[1],x[2]) not in dic:
                dic[(x[0],x[1],x[2])]=[x[3],x[4]]
            else:
                dic[(x[0],x[1],x[2])][0]=list(set(dic[(x[0],x[1],x[2])][0]+x[3]))
                dic[(x[0],x[1],x[2])][1]=sorted(list(set(dic[(x[0],x[1],x[2])][1]+x[4])))
        DFG=[(x[0],x[1],x[2],y[0],y[1]) for x,y in sorted(dic.items(),key=lambda t:t[0][1])]
        return sorted(DFG,key=lambda x:x[1]),states
    else:
        DFG=[]
        for child in root_node.children:
            if child.type in do_first_statement:
                temp,states=DFG_php(child,point2code,states)
                DFG+=temp
        for child in root_node.children:
            if child.type not in do_first_statement:
                temp,states=DFG_php(child,point2code,states)
                DFG+=temp

        return sorted(DFG,key=lambda x:x[1]),states

def DFG_javascript(root_node,point2code,states):
    assignment=['assignment_pattern','augmented_assignment_expression']
    def_statement=['variable_declarator']
    increment_statement=['update_expression']
    if_statement=['if_statement','else']
    for_statement=['for_statement']
    enhanced_for_statement=[]
    while_statement=['while_statement']
    do_first_statement=[]
    states=states.copy()
    if (len(root_node.children)==0 or root_node.type=='string') and root_node.type!='comment':
        idx,code=point2code[(root_node.start_point,root_node.end_point)]
        if root_node.type==code:
            return [],states
        elif code in states:
            return [(code,idx,'comesFrom',[code],states[code].copy())],states
        else:
            if root_node.type=='identifier':
                states[code]=[idx]
            return [(code,idx,'comesFrom',[],[])],states
    elif root_node.type in def_statement:
        name=root_node.child_by_field_name('name')
        value=root_node.child_by_field_name('value')
        DFG=[]
        if value is None:
            indexs=tree_to_variable_index(name,point2code)
            for index in indexs:
                idx,code=point2code[index]
                DFG.append((code,idx,'comesFrom',[],[]))
                states[code]=[idx]
            return sorted(DFG,key=lambda x:x[1]),states
        else:
            name_indexs=tree_to_variable_index(name,point2code)
            value_indexs=tree_to_variable_index(value,point2code)
            temp,states=DFG_javascript(value,point2code,states)
            DFG+=temp            
            for index1 in name_indexs:
                idx1,code1=point2code[index1]
                for index2 in value_indexs:
                    idx2,code2=point2code[index2]
                    DFG.append((code1,idx1,'comesFrom',[code2],[idx2]))
                states[code1]=[idx1]   
            return sorted(DFG,key=lambda x:x[1]),states
    elif root_node.type in assignment:
        left_nodes=root_node.child_by_field_name('left')
        right_nodes=root_node.child_by_field_name('right')
        DFG=[]
        temp,states=DFG_javascript(right_nodes,point2code,states)
        DFG+=temp            
        name_indexs=tree_to_variable_index(left_nodes,point2code)
        value_indexs=tree_to_variable_index(right_nodes,point2code)        
        for index1 in name_indexs:
            idx1,code1=point2code[index1]
            for index2 in value_indexs:
                idx2,code2=point2code[index2]
                DFG.append((code1,idx1,'computedFrom',[code2],[idx2]))
            states[code1]=[idx1]   
        return sorted(DFG,key=lambda x:x[1]),states
    elif root_node.type in increment_statement:
        DFG=[]
        indexs=tree_to_variable_index(root_node,point2code)
        for index1 in indexs:
            idx1,code1=point2code[index1]
            for index2 in indexs:
                idx2,code2=point2code[index2]
                DFG.append((code1,idx1,'computedFrom',[code2],[idx2]))
            states[code1]=[idx1]
        return sorted(DFG,key=lambda x:x[1]),states   
    elif root_node.type in if_statement:
        DFG=[]
        current_states=states.copy()
        others_states=[]
        flag=False
        tag=False
        if 'else' in root_node.type:
            tag=True
        for child in root_node.children:
            if 'else' in child.type:
                tag=True
            if child.type not in if_statement and flag is False:
                temp,current_states=DFG_javascript(child,point2code,current_states)
                DFG+=temp
            else:
                flag=True
                temp,new_states=DFG_javascript(child,point2code,states)
                DFG+=temp
                others_states.append(new_states)
        others_states.append(current_states)
        if tag is False:
            others_states.append(states)        
        new_states={}
        for dic in others_states:
            for key in dic:
                if key not in new_states:
                    new_states[key]=dic[key].copy()
                else:
                    new_states[key]+=dic[key]
        for key in states:
            if key not in new_states:
                new_states[key]=states[key]
            else:
                new_states[key]+=states[key]
        for key in new_states:
            new_states[key]=sorted(list(set(new_states[key])))
        return sorted(DFG,key=lambda x:x[1]),new_states
    elif root_node.type in for_statement:
        DFG=[]
        for child in root_node.children:
            temp,states=DFG_javascript(child,point2code,states)
            DFG+=temp
        flag=False
        for child in root_node.children:
            if flag:
                temp,states=DFG_javascript(child,point2code,states)
                DFG+=temp                
            elif child.type=="variable_declaration":               
                flag=True
        dic={}
        for x in DFG:
            if (x[0],x[1],x[2]) not in dic:
                dic[(x[0],x[1],x[2])]=[x[3],x[4]]
            else:
                dic[(x[0],x[1],x[2])][0]=list(set(dic[(x[0],x[1],x[2])][0]+x[3]))
                dic[(x[0],x[1],x[2])][1]=sorted(list(set(dic[(x[0],x[1],x[2])][1]+x[4])))
        DFG=[(x[0],x[1],x[2],y[0],y[1]) for x,y in sorted(dic.items(),key=lambda t:t[0][1])]
        return sorted(DFG,key=lambda x:x[1]),states
    elif root_node.type in while_statement:  
        DFG=[]
        for i in range(2):
            for child in root_node.children:
                temp,states=DFG_javascript(child,point2code,states)
                DFG+=temp    
        dic={}
        for x in DFG:
            if (x[0],x[1],x[2]) not in dic:
                dic[(x[0],x[1],x[2])]=[x[3],x[4]]
            else:
                dic[(x[0],x[1],x[2])][0]=list(set(dic[(x[0],x[1],x[2])][0]+x[3]))
                dic[(x[0],x[1],x[2])][1]=sorted(list(set(dic[(x[0],x[1],x[2])][1]+x[4])))
        DFG=[(x[0],x[1],x[2],y[0],y[1]) for x,y in sorted(dic.items(),key=lambda t:t[0][1])]
        return sorted(DFG,key=lambda x:x[1]),states    
    else:
        DFG=[]
        for child in root_node.children:
            if child.type in do_first_statement:
                temp,states=DFG_javascript(child,point2code,states)
                DFG+=temp
        for child in root_node.children:
            if child.type not in do_first_statement:
                temp,states=DFG_javascript(child,point2code,states)
                DFG+=temp
        
        return sorted(DFG,key=lambda x:x[1]),states


     
