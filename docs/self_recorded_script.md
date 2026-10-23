# 自录集朗读文本与录音规范（volunteers 集复现）

> 用途：复现论文 §5.1 的 volunteers 数据集（18 人、每人约 10 分钟、办公室底噪环境、1300 条 utterance）。
> 本文本设计为**每位志愿者朗读同一份内容**（约 11 分钟），以便跨说话人做 EER 对比，并为 M4 敏感词检测提供目标。
> **文中所有姓名、电话、住址、证件号、卡号均为虚构，不是任何真实个人信息。**

---

## 一、录音规范（先读完这段再开始）

| 项目 | 要求 |
|---|---|
| 环境 | 普通办公室/宿舍即可，**保留空调声、键盘声等底噪**（论文 volunteers 集就是这个条件）；避免空旷大厅的强回声 |
| 设备 | 手机自带录音 App 或电脑麦克风均可，嘴离麦克风 20–30cm |
| 格式 | 能导出 WAV 最好；手机默认格式（m4a 等）也可以，我们会统一转 16kHz 单声道 WAV |
| 语速 | 自然语速，像平时说话，不要"播音腔"；读错不用倒回去，停顿后把该句重读一遍即可 |
| 分隔 | **每条之间停顿 1–2 秒**（这个停顿很重要，是后续自动切分的依据） |
| 时长 | Part A–D 连续录完约 11 分钟；Part E（中文）可选，再加约 2 分钟 |
| 不适条款 | 个别句子如果让你不舒服（Part C 涉及健康/宗教等虚构内容），可以直接跳过并在交付时说明 |
| 交付 | 一位志愿者一个文件，命名 `p{编号}_{姓名拼音}.wav`（如 `p301_zhangsan.wav`），放到 `E:\project\SpeechGuard-Replica\data\self_recorded\raw\p{编号}\` |

**交付时请同时填写下表（发给收集人即可）：**

| 编号 | 姓名拼音 | 性别 | 年龄 | 母语 | 口音/方言背景 | 完成部分(A-E) |
|---|---|---|---|---|---|---|
| p301 | zhangsan | M | 21 | 中文 | 普通话（北方） | A B C D E |

> 目标人数：组员 2 人必录，另招募 4–8 名志愿者（论文是 18 人，我们尽力即可，报告会写明实际人数）。

---

## Part A · 日常短句（30 条，每条之间停 1–2 秒）

1. The weather today is warmer than yesterday.
2. I take the bus to campus every morning.
3. My favorite season is autumn because of the falling leaves.
4. Coffee helps me stay awake during long study sessions.
5. The library closes at ten o'clock tonight.
6. We watched a documentary about the ocean last weekend.
7. My roommate plays the guitar very well.
8. I need to buy a new pair of running shoes.
9. The train to the city leaves twice every hour.
10. Learning a new language takes patience and practice.
11. Our team won the basketball game by two points.
12. I usually cook noodles when I get home late.
13. The museum opens free of charge on Sundays.
14. My little sister is learning to ride a bicycle.
15. This notebook has survived three semesters of lectures.
16. The mountains look beautiful covered in morning fog.
17. I forgot my umbrella and got soaked on the way home.
18. Reading before bed helps me sleep better.
19. The new coffee shop next to the gate opens tomorrow.
20. Our professor postponed the deadline until Friday.
21. Birds start singing outside my window at dawn.
22. I prefer texting over phone calls.
23. The park is full of children on sunny afternoons.
24. My laptop battery dies within two hours now.
25. We are planning a hiking trip for the holiday.
26. The soup needs a little more salt.
27. I have never been to the western provinces.
28. Street food near the station is cheap and tasty.
29. My grandfather tells the same stories every holiday.
30. A short walk after dinner clears my mind.

---

## Part B · 叙事段落（3 段，每段一口气读完，段内不刻意停顿）

**B1（关于日常）**
My typical day starts at seven in the morning. I grab a quick breakfast, then ride my bike to the lecture hall. Classes keep me busy until noon, and I usually eat lunch with my friends at the canteen. In the afternoon I prefer to study in the library, where the atmosphere is quiet and focused. After dinner, I either go for a jog around the lake or review my notes in the dormitory.

**B2（关于一次旅行）**
Last summer I took a train to the coast with two of my classmates. We left early in the morning and arrived before noon. The seaside town was smaller than we expected, but the beaches were clean and quiet. We rented two bicycles and rode along the shore until sunset. On the last day, we visited an old lighthouse and took dozens of photos. It was a simple trip, but one of the most relaxing weeks I have ever had.

**B3（关于一个爱好）**
I started building model airplanes when I was in primary school. My first model was clumsy, with crooked wings and too much glue. Over the years I learned to sand the frame, balance the wings, and paint thin, even coats. Now I build one model every few months, and each one sits on the shelf above my desk. What I enjoy most is not the finished plane, but the quiet hours spent on each small part.

---

## Part C · 敏感信息句（20 条，**均为虚构信息**，正常语速读出即可）

> 这部分是系统的"测试靶子"：M4 的敏感词检测、定位与加密实验需要文本里真的出现姓名、电话、住址、生日、证件号、健康、宗教、政治等类别（对应论文 §4.3.1 的 GDPR 类词表）。

1. My name is Daniel Reed, and I live at forty-two Orchard Lane, Springfield.
2. You can reach me on my cell phone at five five five, zero one four two.
3. I was born on June the third, nineteen ninety-eight, at City Hospital.
4. My passport number is nine three two four one, and it expires next spring.
5. The password for my email is my daughter's birthday and the word sunshine.
6. Please transfer the rent to my bank card ending in seven seven one nine.
7. Doctor Miller asked me to take one tablet of the blue medicine every morning.
8. I have a cardiology appointment at the clinic on Maple Street this Thursday.
9. My grandmother is diabetic, so we keep insulin in the refrigerator.
10. I attend the community church on River Road every Sunday morning.
11. As a union member, I vote in the factory elections each autumn.
12. My colleague Ahmed and his family moved here from another country two years ago.
13. I support the green party in local elections, though I keep it private.
14. My home address is apartment twelve, ninety-five Birch Avenue, Brookside.
15. The recovery code for my account is four seven two, eight eight, zero six.
16. My ID card was issued last March and carries my biometric data.
17. Miss Chen from human resources has my social insurance number on file.
18. I visit a therapist downtown every other week for my anxiety.
19. Our landlord, Mister Rossi, can be reached at six six six, zero zero two five.
20. The clinic said my blood test results will be ready on Monday morning.

---

## Part D · 数字与编码（10 条）

1. The meeting starts at nine fifteen and ends at noon.
2. My flight number is C A one eight five seven, boarding at gate twenty-two.
3. The total comes to one hundred forty-two dollars and sixty cents.
4. Please enter the verification code: three, nine, one, seven, five.
5. The package will arrive between two and four in the afternoon.
6. Room six oh four is on the sixth floor, next to the elevator.
7. The license plate reads B, seven, K, two, nine, one.
8. We will celebrate my birthday on December the twelfth this year.
9. The discount code is SAVE twenty, valid until October thirty-first.
10. Dial one zero two four for the campus emergency line.

---

## Part E · 中文扩展（10 条，可选）

> 论文语料全是英文；中文部分只用于报告的"扩展讨论"（如中文 ASR/ASV 下的行为差异），不进主实验。

1. 今天下午的会议改到三点半，在二楼会议室开。
2. 我的手机号是一三八零零零零一二三四，有事随时打给我。
3. 我住在向阳路四十二号院五号楼二单元三零一。
4. 我的身份证后四位是六六八八，办卡的时候需要复印件。
5. 周末我常去图书馆自习，人少的时候效率特别高。
6. 医生说我最近需要少熬夜，多喝水，按时吃药。
7. 我奶奶每周日上午都会去社区教堂做礼拜。
8. 快递单号是七七四九零零一一二二，麻烦帮我取一下。
9. 我的邮箱是 zhangsan 二零零一，艾特，q q 点 com。
10. 我们计划五一假期去杭州旅行，车票已经订好了。

---

## 二、这份文本的设计逻辑（给组内人看，志愿者不用读）

| 设计 | 对应实验 |
|---|---|
| Part A/B 所有志愿者读**相同文本** | 跨说话人 EER 计算（同文本不同人，控制变量）；ASR 评估时文本已知可直接算 WER |
| Part A 短句 6–12 词、条间停顿 | 模拟论文 ~8s/条的 utterance 粒度，供 VAD 自动切分 |
| Part B 段落 | 长语音下的 warping 连续性与自然韵律 |
| Part C 敏感类别覆盖：姓名/住址/电话/生日/证件/金融/医疗/宗教/政治/生物识别 | 对齐论文 §4.3.1 的 GDPR 敏感类别与 Table 3 的检测评测；M4 加密定位的靶点 |
| Part D 数字串 | 数字/字母混合内容的对齐与加密帧边界测试 |
| Part E 中文 | 报告"扩展讨论"：英文评估模型（ECAPA/Conformer 均为英文训练）在中文语音上的行为 |
| 录音环境保留底噪 | 复现论文 volunteers 集的 noisy 条件（Table 4），也是加分项"真实场景问题"的素材 |

**处理后命名规则**（收集人/脚本负责）：`{p编号}_{部分}{条目}.wav`，如 `p301_A007.wav`；统一 16kHz 16bit 单声道 WAV，存 `data/self_recorded/processed/`。
