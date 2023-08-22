import Levenshtein
from atypical_settings import *
import re

def correct_consul(message, cname):
    """とりあえず、手書きのまるがついて可笑しくなることが多い'はい'と'いいえ'のみ対応する

    Args:
        message (str): OCR結果の文字列
        cname (str): クラス名

    Returns:
        str: 校正後の文字列
    """

    TEMPLATES = ['はい', 'いいえ']
    NG_WORDS = ['速い', '遅い']

    if not (cname in CONSUL_CNAMES):
        return message

    # remove space
    message = message.replace(' ', '')
    message = message.replace('　', '')
    # 小文字 -> 大文字
    message = message.replace('ぃ', 'い')
    message = message.replace('ぇ', 'え')
    message = message.replace('いいえ', 'いい')
    message = message.replace('いい', 'いいえ')

    message = message.replace('ほい', 'はい')

    for ng in NG_WORDS:
        if ng in message:
            return message

    for tmp in TEMPLATES:
        if tmp in message:
            return tmp

    dists = []

    # 完全一致がないなら、定型分の中で
    # levenが一番短いやつを返す
    for tmp in TEMPLATES:
        dd = Levenshtein.distance(tmp, message)
        # levenがテンプレート文字長よりも長いと好き勝手作れてしまうので
        # その場合はめっちゃ長くする
        if len(tmp) <= dd:
            dd = 9999
        dists.append(dd)

    target = np.argmin(dists)
    # 一番近いものが9999だった場合はそのまま返す
    if dists[target] == 9999:
        return message

    target = TEMPLATES[target]

    return target

def correct_qualitative(message, cname):

    def common(message):

        if message=='':
            return '-'

        tmp = message.replace('(/)', '(+)')
        tmp = tmp.replace('賊', '±')
        tmp = tmp.replace('╋', '+')
        tmp = tmp.replace('一', '-')
        tmp = tmp.replace('━', '-')

        # 半定性値もあるため数字いれる (2+)とか
        pat = re.compile('[1-9+±<>-]')
        chars = pat.findall(tmp)
        if 0 == len(chars):
            return message

        dst = "".join(chars)

        if ('(' in tmp) or (')' in tmp):
            dst = '({})'.format(dst)

        return dst

    def eye_ground(message):
        dst = message.replace('o', '0')
        dst = dst.replace('O', '0')
        dst = dst.replace('Q', '0')
        dst = dst.replace('8', '0')
        dst = dst.replace('(/)', '0')

        dst = dst.replace(':', '')
        dst = dst.replace('・', '')
        dst = dst.replace('-', '')

        if 'l' == dst:
            dst = '1'
        elif 'I' == dst:
            dst = '1'

        return dst

    def blood_type(message):
        tmp = message.replace('0', 'O')
        tmp = tmp.replace('o', 'O')

        dst = ''

        if ('A' in tmp) and ('B' in tmp):
            dst += 'AB'
        elif 'A' in tmp:
            dst += 'A'
        elif 'B' in tmp:
            dst += 'B'
        elif 'O' in tmp:
            dst += 'O'
        else:
            return tmp

        if ('(' in tmp) or (')' in tmp):
            dst = '({})'.format(dst)

        if '型' in tmp:
            dst += '型'

        return dst

    if cname in QUALITATIVE_CNAMES:
        return common(message)
    elif cname == 'btype_abo':
        return blood_type(message)
    elif cname in EYEGROUND_CNAMES:
        return eye_ground(message)
    else:
        return message
### genderの校正をいれる

def correct_other(message, cname):

    if cname == 'examinee_name_kana':
        pat = re.compile('[\u30A1-\u30FF\uFF66-\uFF9F 　]+')
        chars = pat.findall(message)
        return ''.join(chars)
    elif cname == 'examinee_gender':
        dst = ''

        if '男' in message:
            dst += '男'
        elif '女' in message:
            dst += '女'
        else:
            return message

        if '性' in message:
            dst += '性'

        return dst
    else:
        return message

def try_parse(word):
    try:
        dd = float(word)
        return True, dd
    except:
        return False, None

def correct_quantative(message, cname):
    digit_chars = list('0123456789.')

    if cname in QUANTATIVE_CNAMES:
        dst = message.strip()
        dst = dst.replace(',', '.')
        dst = [ww for ww in dst if ww in digit_chars]
        dst = "".join(dst)
        if cname == "eye_sight_cr" or cname == "eye_sight_cl":
            if ('(' in message) or (')' in message):
                dst = '({})'.format(dst)
        return dst
    else:
        return message

def correct_bethesda(message, cname):
    if cname == "cervical_bethesda":
        template=["NILM","ASC-US","ASC-H","LSIL","HSIL","SCC"]
        dists = []
        for tmp in template:
            if tmp in cname:
                return tmp
            dist = Levenshtein.distance(tmp, message)
            if len(tmp) <= dist:
                dist = 9999
            dists.append(dist)

        target = np.argmin(dists)
        # 一番近いものが9999だった場合はそのまま返す
        if dists[target] == 9999:
            return message

        target = template[target]

        return target
    else:
        return message

def verify_che(text, cname):
    che = ['liver_che_ul', 'liver_che_iul']
    thresh_ul = 1500

    if cname in che:
        success, num = try_parse(text)
        if not success:
            return text, cname
        if num < thresh_ul:
            return text, 'liver_che_ul'
        else:
            return text, 'liver_che_iul'
        
    else:
        return text, cname
    
def modify_ocr_text(key, text):
    #key = key.split('-')[0]
    key = key[:-2]
    dst = correct_consul(text, key)
    dst = correct_qualitative(dst, key)
    dst = correct_quantative(dst, key)
    dst = correct_bethesda(dst, key)
    dst = correct_other(dst, key)
    #dst, key = verify_che(dst,key)
    return dst, key

def replace_tougou(key, text):
    tail = key[-2:]
    _key = key[:-2]
    if _key in TOUGOU:
        search_res  = re.findall('[0-9]', text)
        has_scalar  = len(search_res) != 0
        # OCR結果に数字が含まれてるかチェックして、クラス名を書き換える
        # 定性なら(+)とかなので数字は入らない
        if has_scalar == TOUGOU[_key][1]:
            tname = TOUGOU[_key][0]
            tname += tail
            return tname
        
    return key