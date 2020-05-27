## Tourist-Attraction-Alias
The project consists of two documents, one of which is a total of 5,000 manually labeled aliases for tourist attractions(`labeled_data.5k.txt`), and the other is the original data obtained by the reptile, which is unlabeled(`raw_data.txt`). Currently, we open our source for academic study.

&nbsp; 

## Data Format
Each line consists of a title and a text, split by '\t\t':
```
Province#City#Attraction        Introduction text for current attraction
```
For example:
```
四川#甘孜%白玉#白玉偶曲河        如果说[新都桥/SE_False]是[“摄影家走廊”/AE_False]，那么[偶曲河/SE_True]就是[“摄影家的天堂”/AE_True]，甘白路快接近白玉县县城近50公里道路一路沿偶曲河前行，它是从甘孜到白玉的必经之路……
```

&nbsp; 

## Definition
The labeled tag consists of two parts: word property and true/false flag.
### word property:

**AE**: means `alias entity`.It has strong contextual hints, such as:

|Attraction|IntroductionText|
|:--------|:--------|
|故宫|故宫又叫做【紫禁城】… |
|中国国家馆|…后来更名为【中华艺术宫】|
|上海动物园|…原名为【西郊公园】|
|苏州|…有【“人间天堂”】的美誉|

**SE**: means `subject entity`.It usually exist as a subject in a sentence, such as:

|Attraction|IntroductionText|
|:--------|:--------|
|苏州大学|【苏大】校园内风景优美…|
|国家图书馆|【中国国家图书馆】是中国…|
|王府井天主堂|【王府井教堂】位于…|

### true/false flag:
**True**: means the labeled entity is the alias for current attraction.  
**False**: means thelabeled entity is not the alias for current attraction. In other words, it may be alias for other attractions.
