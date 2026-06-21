import re


def process_hallmark_name(name):
    short_words = ('ACID', 'VIA', 'BETA', 'LATE', 'BILE', 'HEME')
    result = []
    words = name.split('_')
    for word in words:
        if re.search(r'\d', word) or len(word) < 5 and word not in short_words:
            result.append(word)
        else:
            result.append(word.lower())
    result = ' '.join(result)
    result = result.replace("TNFA", "TNFα").replace("NFKB", "NFκB").replace("TGF beta", "TGFβ").replace("WNT beta ", "WNT/β-")
    result = result[0].capitalize() + result[1:]
    return result
