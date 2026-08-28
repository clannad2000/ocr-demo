# Beast Academy OCR Demo

这个Demo用于批量处理漫画式数学教材PDF，支持两种模式：

- `exact`：完整OCR，适合建立可审计英文底稿；
- `study`：个人学习翻译，只提取并翻译影响理解的英文自然语言。

基础流程：

1. 使用Poppler将指定PDF页渲染为长边2048像素的PNG。
2. 默认使用DeepSeek-OCR精细定位模式提取文字区域和坐标。
3. 默认使用`qwen3.6-flash`思考模式生成主OCR文本。
4. 运行确定性的漏字、数字和页码风险检查。
5. 默认按策略调用`qwen3.7-max-2026-06-08`非思考模式独立复核。
6. 学习模式通过DeepL Remote MCP翻译选中的英文区域。
7. 输出逐页PNG、JSON、Markdown、批次清单和汇总报告；学习模式另有`study.html`。

复核模型不会自动覆盖或合并主模型文字。

## 运行要求

- Python 3.11或更高版本
- Node.js 18或更高版本
- Poppler命令：`pdfinfo`、`pdftoppm`
- 项目级`@modelcontextprotocol/sdk`（已安装并锁定在`package-lock.json`）
- 可登录的DeepL帐号及有效席位订阅，不需要DeepL API Key

不需要安装额外Python包。如果在新目录中重建项目，运行
`npm install`安装锁定的Node依赖。

程序会自动跳过不兼容的Xpdf版`pdftoppm`，并从`PATH`中选择支持
`-scale-to`和`-singlefile`的Poppler版本。当Xpdf占用`/usr/local/bin/pdftoppm`时，
程序还会检查Homebrew的`/usr/local/opt/poppler/bin/pdftoppm`和
`/opt/homebrew/opt/poppler/bin/pdftoppm`。也可通过
`--pdftoppm-command /absolute/path/to/pdftoppm`显式指定。

## 配置与直接运行

默认读取项目目录中的`ocr_config.json`。配置支持JSONC风格的
`// 单行注释`和`/* 块注释 */`；注释符出现在URL或其他字符串中时不会被删除。
平台密钥只保存在同目录且已被`.gitignore`忽略的`.env`，配置文件中的
`model_profiles`仅定义平台、模型ID、端点和能力，`model_usage`只决定各个步骤
使用哪个资料名。

首次配置时复制`.env.example`为`.env`，然后只在`.env`中填写需要的密钥：

```dotenv
SILICONFLOW_API_KEY=填写SiliconFlow密钥
DASHSCOPE_API_KEY=填写DashScope密钥
DEEPSEEK_API_KEY=如使用DeepSeek官方模型则填写，否则留空
```

若系统环境变量也设置了同名密钥，系统环境变量优先于`.env`。旧配置中的
`api_keys`字段已不再支持；请将其中的值迁移到`.env`后删除该字段。

例如，不改动任何模型资料，只修改主OCR的用途映射：

```json
"model_usage": {
  "layout_ocr": "deepseek-ocr",
  "primary_ocr": "qwen-flash",
  "ocr_verifier": "qwen-max"
}
```

新增任意OpenAI兼容模型时，先在`model_profiles`新增一个唯一资料名，
再在`model_usage`中引用它；切换OCR、翻译复核或整章复核时均无需
复制密钥、端点或模型ID。

不要填写DeepL用户名或密码。然后直接执行：

```bash
cd '/Volumes/DATA2/如子/book/ocr-demo'
python3 ocr_demo.py
```

首次实际翻译会打开DeepL OAuth授权页；授权后，客户端注册信息和令牌保存在
macOS钥匙串项`Beast Academy OCR DeepL MCP`中，项目不会收到或保存
DeepL密码。程序终端输出同时写入配置中的`log_file`。

在不调用任何模型的情况下检查配置：

```bash
python3 ocr_demo.py --dry-run
```

`ocr_config.example.json`是可复制的无密钥模板；`.env.example`是无密钥的
环境变量模板。

## 命令行运行示例

```bash
# 在项目根目录的 .env 中设置密钥（或在系统环境变量中设置）。

python3 ocr_demo.py \
  --pdf '../beast academy math guide 3A.pdf' \
  --pages '25,53,57,73,93' \
  --output './runs/small-batch-01' \
  --workers 2 \
  --verify always \
  --printed-page-offset -1
```


单独跑ocr
```bash
python ocr_demo.py `
  --config .\ocr_config.json `
  --mode exact `
  --verify never `
  --output .\runs\chapter1-exact-01
```

个人学习用的选择性翻译（默认DeepL MCP）：

```bash
python3 ocr_demo.py \
  --pdf '../beast academy math guide 3A.pdf' \
  --pages '25,53,57,73,93' \
  --output './runs/study-batch-01' \
  --mode study \
  --workers 2 \
  --verify auto
```

学习模式会翻译：对白、问题、指令、解释、定义、标题/目录、图注和脚注。
默认跳过：拟声词、独立页脚页码、纯数字表（含百数表）、没有说明文字的公式、
独立的数字几何标注和边长、已划掉的草稿文字。数字或单位如果位于完整问句/解释中，
仍会保留并翻译；目录和索引中的页码/页码范围会保留，装饰性引导点会省略。

正式批量建议使用`--verify auto`。只有在PDF、页码、模型和参数完全相同时，
才应使用`--resume`继续已有任务。

只重做指定异常页，同时复用批次内其他页面：

```bash
python3 ocr_demo.py ... \
  --resume \
  --redo-pages '5,45'
```

术语表或翻译提示词变化后，只重新翻译已有英文区域，不重复图像OCR：

```bash
python3 ocr_demo.py ... \
  --mode study \
  --retranslate
```

只重新翻译少数页时必须显式加载配置。程序会将更新页合并回既有总清单，
不会把`manifest.json`和`study.html`缩成选中的几页：

```bash
python3 ocr_demo.py \
  --config ./ocr_config.json \
  --pages '18,19,23,26,34,40,41' \
  --retranslate
```

DeepL MCP对每个OCR区域单独翻译，避免批量分隔符被改写后导致英中条目错位。
每个区域还会带最多300字符的同页上下文，帮助DeepL正确处理`Acute`、`Right`、
`Sides`这类短标签。远程超时、`fetch failed`和部分5xx错误会有限重试。
翻译若相比原文异常膨胀，会标记`translation_suspicious_expansion`
进入人工复核。

配置中`translation_verify: always`会在每个成功翻译的页面上调用一次
Qwen语义复核。它会检查译义、数学术语、双关语、数字和遗漏，但只产生
`translation_semantic_issue`和建议译文，绝不自动覆盖DeepL结果。如需节省此项
模型调用，可在配置中改为`never`。

如果只需要对已保存的英中译文进行语义复核，不重新OCR、不调用DeepL、
不改动译文，使用：

```bash
python3 ocr_demo.py \
  --config ./ocr_config.json \
  --pages '13-17,20-22,24-25,27-33,35-39,42' \
  --reverify-translation
```

此模式只需`model_usage.translation_verifier`所引用资料的平台密钥，
不会启动DeepL MCP辅助进程。若选中页仍有
`[translation missing]`，程序会停止并提示先运行`--retranslate`。局部复核后仍会
合并回原有总清单和30页`study.html`。
复核模型偶尔可能只给出问题说明而没有建议译文；此类记录仍会保留为人工
复核项，不会导致整页复核结果被丢弃。

### 整章文本盲审

先生成不含图片、不含现有逐页候选问题的盲审Markdown，此步不调用API：

```bash
python3 ocr_demo.py \
  --config ./ocr_config.json \
  --pages '13-42' \
  --export-chapter-review
```

输出`runs/chapter1-study-01/chapter_review_input.md`，其中包含：

- 压缩固定术语表；
- 角色名和专有名称候选；
- 标题/目录对照；
- 重复英文及其当前译法；
- 按页分组的全部英中译文。

确认输入文件后，运行一次整章高上下文盲审：

```bash
caffeinate -i python3 ocr_demo.py \
  --config ./ocr_config.json \
  --pages '13-42' \
  --chapter-review
```

整章复核依次调用`model_usage.chapter_review`选中的文本模型，不调用
OCR或DeepL，也不改动页面JSON和当前译文。例如同时比较两个资料：

```json
"chapter_review": ["qwen-max", "deepseek-v4"]
```

每个模型独立输出：

- `chapter_review-<资料名>-api.json`：完整API响应；
- `chapter_review-<资料名>.json/.md`：验证后的问题和一致性项；
- `chapter_review-comparison.json/.md`：各模型数量与候选区域两两重合情况。

临时覆盖可重复使用`--chapter-review-profile <资料名>`；OCR也可用
`--layout-profile`、`--primary-profile`和`--verifier-profile`临时切换。这些参数
只改变用途选择，不改动`model_profiles`中的模型资料。

### 生成Codex裁决版学习页

当`chapter_review-codex-adjudication.json`已经完成后，可以将其中的明确
替换和全章统一决定应用到内存副本，生成独立学习页：

```bash
python3 ocr_demo.py \
  --config ./ocr_config.json \
  --pages '13-42' \
  --build-reviewed-study
```

输出：

- `study-reviewed.html`：已应用Codex最终译文、术语统一和小学生学习提示；
- `chapter_review-applied.json`：每项原译文、最终译文、决定和理由的应用记录。

该模式不调用OCR、DeepL或任何复核API，也不覆盖页面JSON、
`manifest.json`或原有`study.html`。生成前会核对裁决文件记录的
整章输入、Qwen报告和DeepSeek报告SHA-256；任一源文件发生变化
都会停止应用，避免把旧裁决套用到新译文。

### 生成纯中文版PDF

PDF回填是独立后处理阶段，不重新调用OCR、DeepL、逐页复核或整章模型。
先生成完整且锁定的最终译文快照：

```bash
python3 ocr_demo.py \
  --config ./ocr_config.json \
  --pages '13-42' \
  --build-final-translation
```

`chapter_translation_final.json`以页面JSON中的全部译文为基础，再覆盖Codex
最终裁决；如果配置了人工译文覆盖，则人工结果优先。PDF回填不得绕过该快照
直接使用页面JSON中的旧译文。

生成纯中文版及程序全量技术检查：

```bash
python3 ocr_demo.py \
  --config ./ocr_config.json \
  --pages '13-42' \
  --export-pdf chinese
```

`--export-pdf chinese`会自动重建最终译文快照，因此也可以直接运行第二条命令。
它使用原PDF页面作为底图，按DeepSeek-OCR坐标近似擦除英文并写入
矢量中文。每个OCR坐标只能归属一条译文；存在多个候选气泡时，只使用
匹配度最高的单个空间连通区，宁可保留少量英文，也不跨气泡覆盖。中文字号
按每个文本框独立自适应：先用类型默认字号，文字过多时逐级缩小，短文不放大；
默认最多缩小1点，保留轻微越界以避免中文过小。
宽度最多扩张到原框的1.15倍。允许少量越界、英文残留和背景色差；找不到
可靠坐标的区域保留原文并写入报告，不阻断整章。中文使用独立STHeiti TrueType
字体嵌入，避免苹方TTC在macOS快速查看/PDFKit中出现乱码。
同一条译文的多个英文行框共用一个背景中位色，不再逐行使用不同填充色。
低于65%覆盖率或低于90%的歧义匹配默认保留英文。密集几何图解页默认允许
高置信度定义和图形标签回填，但不因此放宽覆盖率和歧义门槛。

可先只导出代表页做视觉检查：

```bash
python3 ocr_demo.py \
  --config ./ocr_config.json \
  --pages '18,26,28,33,40' \
  --export-pdf chinese
```

子集导出会先在整章范围应用并验证Codex裁决，再只取指定页；输出PDF、快照、
计划和报告的文件名会自动加上`-pages-18_26_28_33_40`，不覆盖整章产物。

主要输出：

- `beast-academy-3A-chapter1-zh-review.pdf`：待人工视觉检查的纯中文版；
- `chapter_translation_final.json`：完整最终译文快照和来源；
- `pdf_backfill_plan.json`：341个区域的坐标、匹配覆盖率和异常；
- `pdf_backfill_report.json`：程序全量结构、计数、字体、坐标唯一归属、文本框严重相交、页面和非空检查；
- `ai_spotcheck_request.json`：只列出建议抽查的少量高风险页，不自动调用AI。

检查顺序固定为“程序全量检查 -> AI可选抽查 -> 人工视觉验收”。程序通过仅表示
`program_checked`，不表示版面美观或人工接受。人工发现问题后，可在配置的局部
版面覆盖文件中记录坐标、字号、颜色或偏移，再只重建PDF；AI不主动逐页检查。

只修改复核规则后，可以直接重新分析已有结果，不会再次调用API：

```bash
python3 ocr_demo.py \
  --pdf '../beast academy math guide 3A.pdf' \
  --pages '25,53,57,73,93' \
  --output './runs/small-batch-01' \
  --verify always \
  --printed-page-offset -1 \
  --reanalyze
```

## 输出结构

```text
runs/small-batch-01/
  manifest.json
  summary.md
  study.html              # 仅学习模式
  pages/
    page-0025.png
    page-0025.json
    page-0025.md
```

纯中文版批次还会在同一输出目录增加最终译文、坐标计划、技术报告、AI抽查请求
和待人工检查PDF；原始PDF及页面JSON保持不变。

JSON包含模型、模式、状态、耗时、Token、OCR原文、风险标记和复核比较；
学习模式还包含逐条英文、中文和已跳过内容，
不保存API密钥、OAuth令牌或请求头。

## 测试

```bash
python3 -m unittest discover -s tests -v
```

## 已验证批次

`runs/study-batch-03`覆盖24个代表页面和359个翻译条目。页面类型包括目录、
索引、普通/黑底漫画、手写笔记、百数表、定义页和几何图。详细评估见
`runs/study-batch-03/evaluation.md`。
