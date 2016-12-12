import re
import pandas as pd

chinese_num = {
    u'一': 1, u'二': 2, u'三': 3, u'四': 4, u'五': 5,
    u'六': 6, u'七': 7, u'八': 8, u'九': 9, u'零': 0,
}


def translate_chinese_number(string):
    string = re.sub('[0-9]+', '', string)
    for k in chinese_num:
        string = string.replace(k, str(chinese_num[k]))   # 十三 -> 十3, 二十一 -> 2十1
    string = re.sub(u'([1-9])十([1-9])', '\\1\\2', string)  # 2十1 -> 21
    string = re.sub(u'([1-9])十', '\g<1>0', string)   # 3十 -> 30
    string = re.sub(u'十([1-9])', '1\\1', string)   # 十3 -> 13
    string = string.replace(u'十', '10')
    string = re.sub(u'([1-9])百([0-9]{2})', '\\1\\2', string)
    string = re.sub(u'([1-9])百([0-9]{1})', '\g<1>\g<2>0', string)
    string = re.sub(u'([1-9])百', '\g<1>00', string)
    nums = re.findall('[0-9]+', string)
    if len(nums) == 0:
        return -1
    return nums[0]


with open('A_LVR_LAND_A.CSV') as f:
    raw = f.read().decode('utf8')
    raw = re.sub('".*?"', '', raw)
    lines = raw.replace('\r', '').split('\n')[:-1]
house = pd.DataFrame([s.split(',') for s in lines[1:]])
house.columns = lines[0].split(',')

house.drop(u'土地區段位置或建物區門牌', axis=1, inplace=True)
house.drop(u'非都市土地使用編定', axis=1, inplace=True)
house.drop(u'非都市土地使用分區', axis=1, inplace=True)
house.drop(u'建物現況格局-衛', axis=1, inplace=True)
house.drop(u'建物現況格局-隔間', axis=1, inplace=True)
house.drop(u'備註', axis=1, inplace=True)
house.drop(u'編號', axis=1, inplace=True)
house.drop(u'車位類別', axis=1, inplace=True)
house.drop(u'交易筆棟數', axis=1, inplace=True)
house.drop(u'車位移轉總面積平方公尺', axis=1, inplace=True)
house.drop(u'車位總價元', axis=1, inplace=True)
house.drop(u'有無管理組織', axis=1, inplace=True)
house.drop(u'主要建材', axis=1, inplace=True)
house.drop(u'主要用途', axis=1, inplace=True)
house.drop(u'建物型態', axis=1, inplace=True)
house.drop(u'總價元', axis=1, inplace=True)

house[u'移轉層次'] = house[u'移轉層次'].apply(translate_chinese_number)
house[u'總樓層數'] = house[u'總樓層數'].apply(translate_chinese_number)
house[u'建築完成年月'][house[u'建築完成年月'] == ''] = '1050101'

for i in range(1, len(house.columns)):
    house[house.columns[i]] = pd.to_numeric(house.iloc[:, i], errors='ignore')
house[u'建築完成年月'] = house[u'建築完成年月'].apply(lambda x: 105 - x // 10000)
house[u'交易年月日'] = house[u'交易年月日'].apply(lambda x: 105 - x // 10000)
house[u'單價每平方公尺'] = house[u'單價每平方公尺'].apply(lambda x: x * 3.306)
house[u'土地移轉總面積平方公尺'] = house[u'土地移轉總面積平方公尺'].apply(lambda x: x / 3.306)
house[u'建物移轉總面積平方公尺'] = house[u'建物移轉總面積平方公尺'].apply(lambda x: x / 3.306)

house = house[house[u'交易標的'] != u'車位']
house = house[house[u'單價每平方公尺'] > 20000]
house = house[house[u'移轉層次'] > 0]
house = house[house[u'總樓層數'] > 0]
house = house.reset_index()

house.to_pickle('taipei_house.pickle')
