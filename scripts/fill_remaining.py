#!/usr/bin/env python3
"""Fill remaining 90 missing translations manually."""
import os, sys, re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

manual = {
    'paintings': '画作',
    'sustained': '持续',
    'adapted': '适应的',
    'accomplished': '有成就的',
    'succeeded': '成功',
    'oriented': '定向的',
    'adjusted': '调整后的',
    'transformed': '转型的',
    'speeds': '速度',
    'drawings': '图纸',
    'strings': '字符串',
    'holdings': '持股',
    'prohibited': '禁止的',
    'ancestors': '祖先',
    'writings': '著作',
    'administered': '管理的',
    'applicants': '申请人',
    'attributes': '属性',
    'interpreted': '解释的',
    'molecules': '分子',
    'listings': '上市',
    'recordings': '录音',
    'readings': '读数',
    'offerings': '祭品',
    'hearings': '听证会',
    'surroundings': '环境',
    'warnings': '警告',
    'biased': '有偏见的',
    'exceeded': '超过',
    'teachings': '教导',
    'weddings': '婚礼',
    'atoms': '原子',
    'emphasized': '强调的',
    'openings': '开口',
    'killings': '杀戮',
    'shootings': '枪击',
    'diminished': '减弱的',
    'meanings': '含义',
    'mornings': '早晨',
    'beginnings': '开始',
    'evenings': '晚上',
    'jose': '何塞',
    'bearings': '轴承',
    'dealings': '交易',
    'deferred': '延期的',
    'endings': '结局',
    'gatherings': '聚会',
    'corpses': '尸体',
    'fiance': '未婚夫',
    'crossings': '交叉口',
    'standings': '排名',
    'ceilings': '天花板',
    'mens': '男式',
    'francois': '弗朗索瓦',
    'denotes': '表示',
    'constrained': '受限的',
    'womens': '女式',
    'rulings': '裁决',
    'conferred': '授予的',
    'dwellings': '住宅',
    'workings': '运作',
    'beyonce': '碧昂丝',
    'landings': '着陆',
    'anecdotes': '轶事',
    'sao': '圣',
    'signings': '签约',
    'fiancee': '未婚妻',
    'andre': '安德烈',
    'synthesized': '合成的',
    'winnings': '赢利',
    'cliche': '陈词滥调',
    'sayings': '谚语',
    'postings': '帖子',
    'somethings': '某些东西',
    'misunderstandings': '误解',
    'happenings': '事件',
    'servings': '份',
    'failings': '缺点',
    'naive': '天真的',
    'childrens': '儿童的',
    'cuttings': '插条',
    'beatings': '殴打',
    'precedes': '先于',
}

# Special: non-ASCII key entries need different handling
special = {
    'jose': 'jos\u00e9',
    'fiance': 'fianc\u00e9',
    'francois': 'fran\u00e7ois',
    'beyonce': 'beyonc\u00e9',
    'fiancee': 'fianc\u00e9e',
    'andre': 'andr\u00e9',
    'sao': 's\u00e3o',
    'cliche': 'clich\u00e9',
    'naive': 'na\u00efve',
}

non_ascii_entries = {
    'jos\u00e9': '何塞',
    'fran\u00e7ois': '弗朗索瓦',
    'beyonc\u00e9': '碧昂丝',
    'fianc\u00e9': '未婚夫',
    'fianc\u00e9e': '未婚妻',
    'andr\u00e9': '安德烈',
    's\u00e3o': '圣',
    'clich\u00e9': '陈词滥调',
    'na\u00efve': '天真的',
    'm\u00fcller': '穆勒',
    'pe\u00f1a': '佩尼亚',
    'garc\u00eda': '加西亚',
    'su\u00e1rez': '苏亚雷斯',
    'mar\u00eda': '玛丽亚',
    'f\u00fcr': '为',
    '\u03bcm': '微米',
}

path = os.path.join(PROJECT_ROOT, 'server', 'translations.py')
with open(path) as f:
    content = f.read()

lines = content.split('\n')
insert_at = None
for i in range(len(lines)-1, -1, -1):
    s = lines[i].strip()
    if s and s[0] == '}':
        insert_at = i
        break

if insert_at:
    # Normal ASCII entries
    new_entries = ['    "{}": "{}",'.format(k, v) for k, v in sorted(manual.items())]
    # Non-ASCII entries
    new_entries += ['    "{}": "{}",'.format(k, v) for k, v in sorted(non_ascii_entries.items())]
    
    new_lines = lines[:insert_at] + new_entries + lines[insert_at:]
    with open(path, 'w') as f:
        f.write('\n'.join(new_lines))
    print('Added {} manual translations'.format(len(manual) + len(non_ascii_entries)))
else:
    print('ERROR: Could not find insertion point')
