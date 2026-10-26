"""生成《志愿者录音指南》PDF（发给志愿者的对外版本）。

内容源自 docs/self_recorded_script.md，做对外改写：
- 删除"设计逻辑"一节、仓库内部路径与 M4/EER 等内部术语；
- 交付方式改为"命名后发送给收集人"（联系方式留待手工填写）；
- 朗读文本 Part A-E 必须逐字保留（全体志愿者读同一份是实验前提）。

运行：python scripts/build_volunteer_guide_pdf.py
输出：docs/志愿者录音指南.pdf
"""
import os
import sys
import tempfile

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    CondPageBreak, HRFlowable, KeepTogether, PageBreak, Paragraph,
    SimpleDocTemplate, Spacer, Table, TableStyle,
)

PDF_SKILL_DIR = r"C:/Users/33588/.zcode/cli/plugins/cache/zcode-plugins-official/document-skills/0.1.4/skills/pdf"
sys.path.insert(0, os.path.join(PDF_SKILL_DIR, "scripts"))
from cover_render import detect_fonts, render_cover  # noqa: E402
from pdf import install_font_fallback  # noqa: E402

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PDF = os.path.join(PROJ, "docs", "志愿者录音指南.pdf")

# ━━ Cascade Palette (pdf.py palette.cascade --mode minimal) ━━
PAGE_BG = colors.HexColor("#f2f3f4")
CARD_BG = colors.HexColor("#ebeef0")
TABLE_STRIPE = colors.HexColor("#ebeeef")
HEADER_FILL = colors.HexColor("#384f5b")
BORDER = colors.HexColor("#a2bac5")
ACCENT = colors.HexColor("#b45536")
TEXT_PRIMARY = colors.HexColor("#202223")
TEXT_MUTED = colors.HexColor("#7f8589")
SEM_WARNING = colors.HexColor("#947841")

# ── Fonts ──
F = r"C:/Windows/Fonts"
pdfmetrics.registerFont(TTFont("SimHei", os.path.join(F, "simhei.ttf")))
pdfmetrics.registerFont(TTFont("YaHei", os.path.join(F, "msyh.ttc"), subfontIndex=0))
pdfmetrics.registerFont(TTFont("YaHei-Bold", os.path.join(F, "msyhbd.ttc"), subfontIndex=0))
pdfmetrics.registerFont(TTFont("Times New Roman", os.path.join(F, "times.ttf")))
pdfmetrics.registerFont(TTFont("Times New Roman Bold", os.path.join(F, "timesbd.ttf")))
registerFontFamily("SimHei", normal="SimHei", bold="SimHei", italic="SimHei")
registerFontFamily("YaHei", normal="YaHei", bold="YaHei-Bold", italic="YaHei")
registerFontFamily("Times New Roman", normal="Times New Roman",
                   bold="Times New Roman Bold", italic="Times New Roman")
install_font_fallback()

CN = "YaHei"
EN = "Times New Roman"

S_H1 = ParagraphStyle("H1", fontName=CN, fontSize=15, leading=21,
                      textColor=HEADER_FILL, spaceBefore=18, spaceAfter=2)
S_H2 = ParagraphStyle("H2", fontName=CN, fontSize=12, leading=16,
                      textColor=TEXT_PRIMARY, spaceBefore=12, spaceAfter=6)
S_BODY = ParagraphStyle("Body", fontName=CN, fontSize=10.5, leading=17,
                        wordWrap="CJK", textColor=TEXT_PRIMARY,
                        alignment=TA_LEFT, spaceAfter=6)
S_BULLET = ParagraphStyle("Bullet", parent=S_BODY, leftIndent=16,
                          bulletIndent=3, spaceAfter=4)
S_READ_EN = ParagraphStyle("ReadEN", fontName=EN, fontSize=11.5, leading=19,
                           leftIndent=26, firstLineIndent=-26,
                           textColor=TEXT_PRIMARY, spaceAfter=5)
S_READ_CN = ParagraphStyle("ReadCN", fontName=CN, fontSize=12, leading=22,
                            leftIndent=26, firstLineIndent=-26,
                            wordWrap="CJK", textColor=TEXT_PRIMARY, spaceAfter=6)
S_PARA_EN = ParagraphStyle("ParaEN", fontName=EN, fontSize=11.5, leading=18,
                           alignment=TA_JUSTIFY, textColor=TEXT_PRIMARY,
                           spaceAfter=8)
S_PARTNOTE = ParagraphStyle("PartNote", fontName=CN, fontSize=9.5, leading=14,
                            wordWrap="CJK", textColor=TEXT_MUTED, spaceAfter=8)
S_CELL = ParagraphStyle("Cell", fontName="SimHei", fontSize=10, leading=14,
                        wordWrap="CJK", textColor=TEXT_PRIMARY)
S_CELL_H = ParagraphStyle("CellH", parent=S_CELL, alignment=TA_CENTER,
                          textColor=colors.white)
S_CAPTION = ParagraphStyle("Caption", fontName=CN, fontSize=9, leading=13,
                           alignment=TA_CENTER, wordWrap="CJK",
                           textColor=TEXT_MUTED, spaceBefore=6, spaceAfter=12)

AVAIL = A4[0] - 2 * 2 * cm  # 页宽减左右边距


def h1(num, title):
    return [
        CondPageBreak(140),
        Paragraph(f"<b>{num}　{title}</b>", S_H1),
        HRFlowable(width="100%", thickness=1.4, color=ACCENT,
                   spaceBefore=2, spaceAfter=10),
    ]


def bullet(text):
    return Paragraph(text, S_BULLET, bulletText="•")


def numbered(n, text, style):
    return Paragraph(f"{n}.  {text}", style)


def table(data, ratios, caption=None, header=True):
    widths = [r * AVAIL for r in ratios]
    assert sum(widths) <= AVAIL + 0.5
    t = Table(data, colWidths=widths, hAlign="CENTER", repeatRows=1 if header else 0)
    cmds = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
    ]
    if header:
        cmds += [("BACKGROUND", (0, 0), (-1, 0), HEADER_FILL)]
        for i in range(1, len(data)):
            if i % 2 == 0:
                cmds.append(("BACKGROUND", (0, i), (-1, i), TABLE_STRIPE))
    t.setStyle(TableStyle(cmds))
    out = [t]
    if caption:
        out.append(Paragraph(caption, S_CAPTION))
    else:
        out.append(Spacer(1, 12))
    return out


def callout(text, border_color=ACCENT, bg=CARD_BG):
    p = Paragraph(text, ParagraphStyle("CO", parent=S_BODY, fontSize=10,
                                       leading=16, spaceAfter=0))
    t = Table([[p]], colWidths=[AVAIL * 0.96], hAlign="CENTER")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("LINEBEFORE", (0, 0), (0, -1), 3, border_color),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
    ]))
    return [Spacer(1, 4), t, Spacer(1, 12)]


# ── 朗读文本（逐字来自 docs/self_recorded_script.md，不得改动）──
PART_A = [
    "The weather today is warmer than yesterday.",
    "I take the bus to campus every morning.",
    "My favorite season is autumn because of the falling leaves.",
    "Coffee helps me stay awake during long study sessions.",
    "The library closes at ten o'clock tonight.",
    "We watched a documentary about the ocean last weekend.",
    "My roommate plays the guitar very well.",
    "I need to buy a new pair of running shoes.",
    "The train to the city leaves twice every hour.",
    "Learning a new language takes patience and practice.",
    "Our team won the basketball game by two points.",
    "I usually cook noodles when I get home late.",
    "The museum opens free of charge on Sundays.",
    "My little sister is learning to ride a bicycle.",
    "This notebook has survived three semesters of lectures.",
    "The mountains look beautiful covered in morning fog.",
    "I forgot my umbrella and got soaked on the way home.",
    "Reading before bed helps me sleep better.",
    "The new coffee shop next to the gate opens tomorrow.",
    "Our professor postponed the deadline until Friday.",
    "Birds start singing outside my window at dawn.",
    "I prefer texting over phone calls.",
    "The park is full of children on sunny afternoons.",
    "My laptop battery dies within two hours now.",
    "We are planning a hiking trip for the holiday.",
    "The soup needs a little more salt.",
    "I have never been to the western provinces.",
    "Street food near the station is cheap and tasty.",
    "My grandfather tells the same stories every holiday.",
    "A short walk after dinner clears my mind.",
]
PART_B = [
    ("B1　关于日常",
     "My typical day starts at seven in the morning. I grab a quick breakfast, "
     "then ride my bike to the lecture hall. Classes keep me busy until noon, and "
     "I usually eat lunch with my friends at the canteen. In the afternoon I "
     "prefer to study in the library, where the atmosphere is quiet and focused. "
     "After dinner, I either go for a jog around the lake or review my notes in "
     "the dormitory."),
    ("B2　关于一次旅行",
     "Last summer I took a train to the coast with two of my classmates. We left "
     "early in the morning and arrived before noon. The seaside town was smaller "
     "than we expected, but the beaches were clean and quiet. We rented two "
     "bicycles and rode along the shore until sunset. On the last day, we visited "
     "an old lighthouse and took dozens of photos. It was a simple trip, but one "
     "of the most relaxing weeks I have ever had."),
    ("B3　关于一个爱好",
     "I started building model airplanes when I was in primary school. My first "
     "model was clumsy, with crooked wings and too much glue. Over the years I "
     "learned to sand the frame, balance the wings, and paint thin, even coats. "
     "Now I build one model every few months, and each one sits on the shelf "
     "above my desk. What I enjoy most is not the finished plane, but the quiet "
     "hours spent on each small part."),
]
PART_C = [
    "My name is Daniel Reed, and I live at forty-two Orchard Lane, Springfield.",
    "You can reach me on my cell phone at five five five, zero one four two.",
    "I was born on June the third, nineteen ninety-eight, at City Hospital.",
    "My passport number is nine three two four one, and it expires next spring.",
    "The password for my email is my daughter's birthday and the word sunshine.",
    "Please transfer the rent to my bank card ending in seven seven one nine.",
    "Doctor Miller asked me to take one tablet of the blue medicine every morning.",
    "I have a cardiology appointment at the clinic on Maple Street this Thursday.",
    "My grandmother is diabetic, so we keep insulin in the refrigerator.",
    "I attend the community church on River Road every Sunday morning.",
    "As a union member, I vote in the factory elections each autumn.",
    "My colleague Ahmed and his family moved here from another country two years ago.",
    "I support the green party in local elections, though I keep it private.",
    "My home address is apartment twelve, ninety-five Birch Avenue, Brookside.",
    "The recovery code for my account is four seven two, eight eight, zero six.",
    "My ID card was issued last March and carries my biometric data.",
    "Miss Chen from human resources has my social insurance number on file.",
    "I visit a therapist downtown every other week for my anxiety.",
    "Our landlord, Mister Rossi, can be reached at six six six, zero zero two five.",
    "The clinic said my blood test results will be ready on Monday morning.",
]
PART_D = [
    "The meeting starts at nine fifteen and ends at noon.",
    "My flight number is C A one eight five seven, boarding at gate twenty-two.",
    "The total comes to one hundred forty-two dollars and sixty cents.",
    "Please enter the verification code: three, nine, one, seven, five.",
    "The package will arrive between two and four in the afternoon.",
    "Room six oh four is on the sixth floor, next to the elevator.",
    "The license plate reads B, seven, K, two, nine, one.",
    "We will celebrate my birthday on December the twelfth this year.",
    "The discount code is SAVE twenty, valid until October thirty-first.",
    "Dial one zero two four for the campus emergency line.",
]
PART_E = [
    "今天下午的会议改到三点半，在二楼会议室开。",
    "我的手机号是一三八零零零零一二三四，有事随时打给我。",
    "我住在向阳路四十二号院五号楼二单元三零一。",
    "我的身份证后四位是六六八八，办卡的时候需要复印件。",
    "周末我常去图书馆自习，人少的时候效率特别高。",
    "医生说我最近需要少熬夜，多喝水，按时吃药。",
    "我奶奶每周日上午都会去社区教堂做礼拜。",
    "快递单号是七七四九零零一一二二，麻烦帮我取一下。",
    "我的邮箱是 zhangsan 二零零一，艾特，q q 点 com。",
    "我们计划五一假期去杭州旅行，车票已经订好了。",
]

REG_SPEC = [
    ("环境", "普通办公室或宿舍即可，保留空调声、键盘声等日常底噪（本研究特意需要带真实环境音的语音）；避免空旷大厅的强回声。"),
    ("设备", "手机自带录音 App 或电脑麦克风均可；嘴巴离麦克风约 20–30 厘米。"),
    ("格式", "能导出 WAV 最好；手机默认格式（m4a 等）也可以，我们会统一转换格式。"),
    ("语速", "自然语速，像平时说话，不要用「播音腔」；读错不用倒回去，停顿后把该句重读一遍即可。"),
    ("句间停顿", "每条之间停顿 1–2 秒。这个停顿非常重要，是后续自动切分句子的依据。"),
    ("时长", "Part A–D 连续录完约 11 分钟；Part E（中文，可选）再加约 2 分钟。"),
    ("跳过条款", "个别句子如果让你不舒服（Part C 涉及健康、宗教等虚构内容），可以直接跳过，交付时说明即可。"),
    ("交付", "一位志愿者一个文件，命名为 p{编号}_{姓名拼音}.wav（如 p301_zhangsan.wav），发送给收集人（联系方式见文末）。"),
]


def build_body(path):
    doc = SimpleDocTemplate(
        path, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2.2 * cm, bottomMargin=2 * cm,
        title="志愿者录音指南", author="SpeechGuard-Replica 项目组",
        subject="语音隐私保护研究项目音频数据采集 · 志愿者录音说明与朗读文本",
    )

    def footer(canv, d):
        canv.saveState()
        canv.setStrokeColor(BORDER)
        canv.setLineWidth(0.5)
        y = 1.15 * cm
        canv.line(2 * cm, y, A4[0] - 2 * cm, y)
        canv.setFont(CN, 8.5)
        canv.setFillColor(TEXT_MUTED)
        canv.drawString(2 * cm, y - 14, "SpeechGuard 语音隐私保护研究 · 志愿者录音指南")
        canv.drawRightString(A4[0] - 2 * cm, y - 14, f"第 {d.page} 页")
        canv.restoreState()

    story = []

    # ── 一、关于本次录音 ──
    story += h1("一", "关于本次录音")
    story.append(Paragraph(
        "我们是《语音识别与合成》课程的项目组，正在复现 USENIX Security 2025 论文 "
        "<font name=\"Times New Roman Bold\">SpeechGuard</font>（一种可恢复、可定制的语音隐私保护系统）。"
        "论文的全部实验建立在 18 名志愿者朗读的自建数据集之上。要让系统在我们自己的数据上同样成立，"
        "我们需要一批条件相近的朗读语音，因此诚挚邀请你参与这次录音：朗读一份统一准备好的文本，"
        "全程约 11 分钟。", S_BODY))
    story += callout(
        "<b>三点说明</b><br/>"
        "1. 所有志愿者朗读<b>同一份文本</b>（第三部分），这是跨说话人对比实验的前提，请勿自行增删改；<br/>"
        "2. 文本中出现的一切姓名、电话、住址、证件号<b>均为虚构</b>，不包含任何真实个人信息；<br/>"
        "3. 你的录音与登记信息<b>仅用于本课程研究</b>，不公开发布、不提供任何第三方；"
        "你可以随时退出并要求删除录音。")
    story.append(Paragraph(
        "录音不需要任何专业设备，也不需要安静环境——真实、日常的声音正是我们想要的样本。"
        "开始前请先读完第二节的规范，然后打开录音 App，按第三部分从 Part A 顺序读到 Part D"
        "（Part E 为可选的中文部分）。最后按第四节的命名方式发回给我们即可。", S_BODY))

    # ── 二、录音规范 ──
    story += h1("二", "录音规范")
    data = [[Paragraph("<b>项目</b>", S_CELL_H), Paragraph("<b>要求</b>", S_CELL_H)]]
    data += [[Paragraph(k, S_CELL), Paragraph(v, S_CELL)] for k, v in REG_SPEC]
    story += table(data, [0.17, 0.83],
                   caption="表 1　录音规范一览（先读完再开始）")

    # ── 三、朗读文本 ──
    story += h1("三", "朗读文本")
    story.append(Paragraph(
        "以下即需要朗读的全部内容，共五个部分。英文句请正常发音、正常语速；"
        "数字已按口语习惯拼写，照着读即可。建议分段录制也可以，但请确保一条不缺、句间有停顿。",
        S_BODY))

    story.append(Paragraph("<b>Part A · 日常短句</b>（30 条，每条之间停 1–2 秒）", S_H2))
    story.append(Paragraph("热身的英文陈述句，读起来顺畅后再进入后面的部分。", S_PARTNOTE))
    story += [numbered(i + 1, s, S_READ_EN) for i, s in enumerate(PART_A)]

    story.append(CondPageBreak(160))
    story.append(Paragraph("<b>Part B · 叙事段落</b>（3 段，每段一口气读完，段内不刻意停顿）", S_H2))
    story.append(Paragraph("考察长语音下的自然韵律：像讲故事一样连贯地读完一整段。", S_PARTNOTE))
    for title, para in PART_B:
        story.append(KeepTogether([
            Paragraph(f"<b>{title}</b>",
                      ParagraphStyle("Bt", parent=S_BODY, fontName=CN,
                                     textColor=HEADER_FILL, spaceAfter=4)),
            Paragraph(para, S_PARA_EN)]))

    story.append(CondPageBreak(160))
    story.append(Paragraph("<b>Part C · 敏感信息句</b>（20 条，均为虚构信息，正常语速读出即可）", S_H2))
    story += callout(
        "下面 20 句中的姓名、电话、住址、证件号、卡号等<b>全部是虚构的</b>，不包含任何真实个人信息。"
        "它们是系统需要自动识别并加密保护的「敏感信息」样本，请像读普通句子一样自然读出来。",
        border_color=SEM_WARNING, bg=PAGE_BG)
    story += [numbered(i + 1, s, S_READ_EN) for i, s in enumerate(PART_C)]

    story.append(CondPageBreak(160))
    story.append(Paragraph("<b>Part D · 数字与编码</b>（10 条）", S_H2))
    story.append(Paragraph("数字串与字母串按写法逐字读出，不必读成数学表达。", S_PARTNOTE))
    story += [numbered(i + 1, s, S_READ_EN) for i, s in enumerate(PART_D)]

    story.append(CondPageBreak(160))
    story.append(Paragraph("<b>Part E · 中文扩展</b>（10 条，可选）", S_H2))
    story.append(Paragraph("这部分用于报告的扩展讨论（系统对非英语语音的表现），不进主实验；愿意多帮忙的话请一并录制。", S_PARTNOTE))
    story += [numbered(i + 1, s, S_READ_CN) for i, s in enumerate(PART_E)]

    # ── 四、交付与登记 ──
    story += h1("四", "交付与登记")
    story.append(Paragraph(
        "录完后请把音频文件重命名为 p{编号}_{姓名拼音}.wav（例如 p301_zhangsan.wav）。"
        "你的编号由收集人分配。请把录音文件和下面这张登记表一起发送给收集人。", S_BODY))
    cols = ["编号", "姓名拼音", "性别", "年龄", "母语", "口音/方言背景", "完成部分(A–E)"]
    ratios = [0.11, 0.19, 0.08, 0.08, 0.11, 0.24, 0.19]
    data = [[Paragraph(f"<b>{c}</b>", S_CELL_H) for c in cols]]
    example = ["p301", "zhangsan", "M", "21", "中文", "普通话（北方）", "A B C D E"]
    data.append([Paragraph(f"{c}", ParagraphStyle("ex", parent=S_CELL,
                                                  textColor=TEXT_MUTED))
                 for c in example])
    for _ in range(3):
        data.append([Paragraph("", S_CELL) for _ in cols])
    story += table(data, ratios,
                   caption="第 2 行为示例（灰色），请在其余行填入你的信息；行不够可自行复制添加。")
    story += callout(
        "<b>收集人</b>：＿＿＿＿＿＿＿＿　　<b>联系方式</b>（微信 / 邮箱）：＿＿＿＿＿＿＿＿＿＿＿＿<br/>"
        "<b>交付截止日期</b>：＿＿＿＿＿＿＿＿", bg=PAGE_BG, border_color=HEADER_FILL)
    story.append(Paragraph(
        "最后，真的非常感谢！这批语音是整个研究能不能落地的关键：没有真实自录数据，"
        "系统效果就只能停留在论文的数字里。如果你在录制中遇到任何不清楚的地方，"
        "随时联系收集人。—— SpeechGuard 项目组", S_BODY))

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main():
    tmp = tempfile.mkdtemp(prefix="volguide_")
    body = os.path.join(tmp, "body.pdf")
    cover = os.path.join(tmp, "cover.pdf")
    build_body(body)

    fonts = detect_fonts()
    palette = {"primary": "#384f5b", "secondary": "#556973",
               "text": "#202223", "muted": "#7f8589", "bg": "#ffffff"}
    content = {
        "kicker": "语音隐私保护研究项目 · VOICE DATA COLLECTION",
        "hero": "志愿者录音指南",
        "summary": "诚邀你为一项语音隐私保护研究录制约 11 分钟的朗读语音。"
                   "全部文本中的姓名、电话、住址均为虚构；录音仅用于课程研究，可随时撤回删除。"
                   "开始录音前，请完整阅读第二节的录音规范。",
        "meta": "SpeechGuard-Replica 项目组 · 2026 年秋季学期",
        "footer": "SPEECHGUARD VOLUNTEER GUIDE",
        "year": "2026",
        "word": "VOICE",
    }
    render_cover("04", content, cover, palette=palette, fonts=fonts)

    from pypdf import PdfReader, PdfWriter
    w = PdfWriter()
    w.add_page(PdfReader(cover).pages[0])
    for p in PdfReader(body).pages:
        w.add_page(p)
    w.add_metadata({
        "/Title": "志愿者录音指南",
        "/Author": "SpeechGuard-Replica 项目组",
        "/Subject": "语音隐私保护研究项目音频数据采集",
        "/Creator": "SpeechGuard-Replica",
    })
    with open(OUT_PDF, "wb") as f:
        w.write(f)
    n = len(PdfReader(OUT_PDF).pages)
    print(f"OK -> {OUT_PDF}  pages={n}")


if __name__ == "__main__":
    main()
