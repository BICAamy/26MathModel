from matplotlib import font_manager

# 列出所有字体名里带常见中文字体关键词的
for f in font_manager.fontManager.ttflist:
    name = f.name
    if any(k in name for k in ["Hei", "Song", "Kai", "PingFang", "STHeiti",
                               "Songti", "Heiti", "Hiragino", "Yuanti",
                               "WenQuanYi", "Noto Sans CJK", "SimHei"]):
        print(name, "->", f.fname)