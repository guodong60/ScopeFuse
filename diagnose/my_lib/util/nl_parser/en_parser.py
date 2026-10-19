#coding=utf-8
import nltk
import re
from nltk.tokenize import MWETokenizer
from nltk import WordPunctTokenizer
import string
from copy import deepcopy
# from nltk.corpus import wordnet
import enchant

punc_str2="""\\‖`§！？｡＂＃＄％＆＇（）＊＋－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏"""
punc_str=string.punctuation+punc_str2
punc_str=punc_str.replace('_','')
# list_tuple=List[tuple]

#一些特殊缩写，nltk无法扩展分割，使用字典的方式记录供使用
def get_sp_abbr_dict():
    sp_abbr_dict={"don't":"do not","doesn't":"does not","didn't":"did not","haven't":"have not","hasn't":"has not","hadn't":"had not","isn't":"is not","wasn't":"was not","weren't":"were not","aren't":"are not","can't":"can not","couldn't":"could not","shouldn't":"should not","wouldn't":"would not","mustn't":"must not","mightn't":"might not","won't":"will not","shan't":"shall not","he's":"he is","she's":"she is","it's":"it is","that's":"that is","let's":"let us","there's":"there is","here's":"here is","what's":"what is","how's":"how is","who's":"who is","where's":"where is","I'm":"I am"}
    nsp_abbr_dict=dict()
    for abbr,nt in sp_abbr_dict.items():
        nsp_abbr_dict[abbr.upper()]=nt.upper()
        nsp_abbr_dict[abbr[0].upper()+abbr[1:]]=nt[0].upper()+nt[1:]
        if abbr.endswith("'t") or abbr.endswith("'s") and abbr!="it's":
            wabbr=abbr.replace("'",'')
            nsp_abbr_dict[wabbr]=nt
            nsp_abbr_dict[wabbr.upper()]=nt.upper()
            nsp_abbr_dict[wabbr[0].upper()+wabbr[1:]]=nt[0].upper()+nt[1:]

    sp_abbr_dict2={"'re":'are',"'ve":"have","'ll":"will"}
    nsp_abbr_dict2=dict()
    for abbr in sp_abbr_dict2.keys():
        nsp_abbr_dict2[abbr.upper()]=sp_abbr_dict2[abbr].upper()
    sp_abbr_dict={**sp_abbr_dict,**nsp_abbr_dict,**sp_abbr_dict2,**nsp_abbr_dict2}
    # nsp_abbr_dict=dict()
    # for key in sorted(sp_abbr_dict.keys()):
    #     nsp_abbr_dict[key]=sp_abbr_dict[key]
    return sp_abbr_dict
SP_ABBR_DICT=get_sp_abbr_dict()


class EnWordCheck(object):
    def __init__(self,user_words=None, exclude_words=None):
        '''

        :param user_words: all user words must be lowercased
        :param exclude_words:
        '''
        self.user_words = set(user_words) if user_words else set()
        self.exclude_words = set(exclude_words) if exclude_words else set()
        # self.dynamic_vocab = set()  # 动态词库会在分割时动态更新，减少eng_dict.check的时间
        self.en_uk_dict = enchant.Dict("en_UK")  # 初始化一个UK检测器
        self.en_us_dict = enchant.Dict("en_US")  # 初始化一个US检测器

    def check(self, string):
        '''
        判断string是否为英文单词
        :param string:
        :return:
        '''
        string=string.lower()
        try:
            # if string in self.dynamic_vocab or \
            #         (string not in self.exclude_words and
            #          (self.en_us_dict.check(string) or self.en_uk_dict.check(string) or
            #           (len(string) > 2 and wordnet.synsets(string)))):
            if string in self. user_words or \
                    (string not in self.exclude_words and
                     (self.en_us_dict.check(string) or self.en_uk_dict.check(string))):
                # self.dynamic_vocab.add(string)
                return True
            return False
        except Exception:
            return False

def tokenize_english(text,vocabs=None,keep_punc=True,keep_stopword=True,lemmatize=True,lower=True,punc_str=punc_str,
                     user_words=None):
    '''

    :param text:
    :param vocabs:
    :param keep_punc:
    :param keep_stopword:
    :param lemmatize:
    :param lower:
    :param punc_str:
    :param user_words: List[tuple]
    :return:
    '''
    # if lower:
    #     text=text.lower()
    user_words=[] if user_words is None else user_words
    text=text.replace("``",'"').replace("''",'"').replace('`',"'")
    text=re.sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*",r'<url>',text,re.S)

    text=' - '.join(text.split('-'))

    puncs = re.findall(r'[{}]+'.format(punc_str), text, re.S)
    diff_puncs = filter(lambda x: len(x) > 1 and len(set(x)) > 1, puncs)
    same_puncs = filter(lambda x: len(x) > 1 and len(set(x)) == 1, puncs)
    for punc in diff_puncs:
        text = text.replace(punc, ' ' + ' '.join(punc) + ' ')
    for punc in same_puncs:
        text = text.replace(punc, ' ' + punc[0] + ' ')

    tokenizer=MWETokenizer([('<','url','>')]+user_words,separator='')
    # nl_parser.add_mwe(user_words)
    # text=text.replace('- -','--')
    words=tokenizer.tokenize(nltk.wordpunct_tokenize(text))
    if lemmatize:
        lemmatizer = nltk.stem.WordNetLemmatizer()  # 词干提取
        if vocabs is not None:
            words = [lemmatizer.lemmatize(lemmatizer.lemmatize(lemmatizer.lemmatize(word,pos='n'),pos='v'),pos='a')
                     if word not in vocabs else word for word in words]
        else:
            words = [lemmatizer.lemmatize(lemmatizer.lemmatize(lemmatizer.lemmatize(word,pos='n'),pos='v'),pos='a')
                     for word in words]

    if not keep_punc:
        stop_puncs_0 = ['|', '{}', '()', '[]', '&', '*',
                        '/', '//', '#', '\\', '~', '""', '‖', '§']
        stop_puncs_1 = ['、', '\'', '"', '.', ':', ',', '...', '{', '}', '(', ')', '[', ']',
                        ';', '?', '!', '-', '--']
        # stop_puncs_2 = ["``", "''",'`']
        stop_puncs = stop_puncs_0 + stop_puncs_1
        words=[word for word in words if word not in stop_puncs]
    if not keep_stopword:
        stop_words = nltk.corpus.stopwords.words('english')
        words=[word for word in words if word not in stop_words]
    if lower:
        if vocabs is not None:
            words=[word.lower() if word not in vocabs else word for word in words]
        else:
            words=[word.lower() for word in words]
    text=' '.join(words)

    #针对这些分词后(doesn't hasn't isn't weren't aren't didn't don't won't wouldn't shouldn't 've 'd 'm 's)
    abbs=[(" ' t "," 't "),(" ' d "," 'd "),(" ' m "," 'm "),(" ' s "," 's "),(" ' ve "," 've ")]
    for str1,str2 in abbs:
        text=text.replace(str1,str2)
    words=text.split()
    return words

def tokenize_glove(text,vocabs=None,keep_punc=True,keep_stopword=True,lemmatize=True,lower=False):
    '''
    针对glove词库做分词
    :param text:
    :param vocabs: glove字典,glove字典中词语区分大小写
    :param keep_punc:
    :param keep_stopword:
    :param lemmatize:
    :param lower:
    :return:
    '''
    # stop_puncs=['"','.',':',',','...','|','{','}','{}','(',')','()','[',']','[]','&','*','`',
    #             '/','//','#','\\','~','、',';','?','!','\'','-','--','""','‖','§']

    # stemmer=nltk.stem.SnowballStemmer(language='english')
    # lemmatizer = nltk.stem.WordNetLemmatizer()  #词干提取
    # text = (' ' + text + ' ').lower()
    # if text.startwith('\''):
    #     text=text.lstrip('\'')
    #     text='\' '+text
    # eyes = "[8:=;]"
    # nose = "['`\-]?"
    text=text.replace("``",'"').replace("''",'"').replace('`',"'")
    text=re.sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*",r'<url>',text,re.S)

    text=re.sub(r"#{[8:=;]}#{['`\-]?}[)d]+|[)d]+#{['`\-]?}#{[8:=;]}",'<smile>',text,re.S)
    text = re.sub(r"#{[8:=;]}#{['`\-]?}p+", '<lolface>', text,re.S)
    text = re.sub(r"#{[8:=;]}#{['`\-]?}\(+|\)+#{['`\-]?}#{[8:=;]}", '<sadface>', text, re.S)
    text = re.sub(r"#{[8:=;]}#{['`\-]?}[\/|l*]", '<neutralface>', text, re.S)
    text = re.sub(r"<3", '<heart>', text,re.S)
    text = re.sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", '<number>', text, re.S)

    text = (' ' + text).replace(' \'',' \' ').strip()
    # text = (' ' + text + ' ').lower().lstrip(' \'').rstrip('\' ').strip()
    text=' - '.join(text.split('-'))
    # text=text.replace('- -','--')
    words=nltk.word_tokenize(text)
    if lemmatize:
        lemmatizer = nltk.stem.WordNetLemmatizer()  # 词干提取

        if vocabs is not None:
            words = [lemmatizer.lemmatize(lemmatizer.lemmatize(lemmatizer.lemmatize(word,pos='n'),pos='v'),pos='a')
                     if word not in vocabs else word for word in words]

        else:
            words = [lemmatizer.lemmatize(lemmatizer.lemmatize(lemmatizer.lemmatize(word,pos='n'),pos='v'),pos='a')
                     for word in words]
    words = ' '.join(words).replace('< url >', '<url>').replace('< smile >', '<smile>') \
        .replace('< lolface >', '<lolface>').replace('< sadface >', '<sadface>') \
        .replace('< neutralface >', '<neutralface>').replace('< heart >', '<v>') \
        .replace('< number >', '<number>').split()
    # words = text.split()
    if not keep_punc:
        stop_puncs_0 = ['|', '{}', '()', '[]', '&', '*',
                        '/', '//', '#', '\\', '~', '""', '‖', '§']
        stop_puncs_1 = ['、', '\'', '"', '.', ':', ',', '...', '{', '}', '(', ')', '[', ']',
                        ';', '?', '!', '-', '--']
        # stop_puncs_2 = ["``", "''",'`']
        stop_puncs = stop_puncs_0 + stop_puncs_1
        words=[word for word in words if word not in stop_puncs]
    if not keep_stopword:
        stop_words = nltk.corpus.stopwords.words('english')
        words=[word for word in words if word not in stop_words]
    if lower:
        if vocabs is not None:
            words=[word.lower() if word not in vocabs else word for word in words]
        else:
            words=[word.lower() for word in words]
    return words

