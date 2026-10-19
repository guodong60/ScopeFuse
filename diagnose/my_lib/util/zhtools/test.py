from langconv import Converter #

def cat_to_chs(sentence): #传入参数为列表
        """
        将繁体转换成简体
        :param line:
        :return:
        """
        sentence = Converter('zh-hans').convert(sentence)

        return sentence


def chs_to_cht(sentence):#传入参数为列表
        """
        将简体转换成繁体
        :param sentence:
        :return:
        """
        sentence = Converter('zh-hant').convert(sentence)

        return sentence

if __name__=='__main__':
    text='寶寶的種族'
    print(cat_to_chs(text))