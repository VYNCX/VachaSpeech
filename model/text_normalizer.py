import re
from pythainlp.util import num_to_thaiword,normalize,maiyamok

def number_to_text(text):
    pattern = r"([-+]?\d*\.\d+|\d+)"
    def replacer(match):
        num_str = match.group(0)
        try:
            if '.' in num_str:
                integer_part, decimal_part = num_str.split('.')
                integer_word = num_to_thaiword(int(integer_part))
                decimal_word = ''.join([num_to_thaiword(int(d)) for d in decimal_part])
                thai_word = f"{integer_word}จุด{decimal_word}"
            else:
                thai_word = num_to_thaiword(int(num_str))
            return thai_word
        except Exception as e:
            return num_str
    return re.sub(pattern, replacer, text)

def normalize_text(text:str):
    text = text.replace(" ๆ","ๆ")
    text_number = number_to_text(text)
    text_cleaned = maiyamok(normalize(text_number))
    return "".join(text_cleaned)