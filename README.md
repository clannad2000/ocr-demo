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
- 项目级`@modelcontextprotocol/sdk`（已锁定在`package-lock.json`）
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
# 首次DeepL OAuth登录后由程序自动生成，请勿手工填写或提交
DEEPL_OAUTH_CREDENTIALS=
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
# macOS / Linux
cd '/Volumes/DATA2/如子/book/ocr-demo'
python3 ocr_demo.py
```

```powershell
# Windows PowerShell
Set-Location 'D:\project\ocr-demo'
python .\ocr_demo.py
```

首次实际翻译会打开DeepL OAuth授权页；授权后，客户端注册信息和令牌会编码
保存到项目`.env`的`DEEPL_OAUTH_CREDENTIALS`，后续刷新令牌时自动更新该值。
项目不会收到或保存DeepL密码，也不会把OAuth凭据写入日志。请像保护API Key
一样保护`.env`，不要提交或分享。程序终端输出同时写入配置中的`log_file`。

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

个人学习用的选择性翻译,带ocr（默认DeepL MCP）：

```bash
 python ocr_demo.py   --pdf './book/beast academy math guide 3A.pdf'   --pages '25,53'   --output './runs/study-batch-01'   --mode study   --workers 2   --verify auto
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

DeepL翻译后的语义复核默认关闭，配置文件使用`translation_verify`控制：
`never`不调用复核模型，`always`会在每个成功翻译的页面上调用一次Qwen语义复核。
语义复核会检查译义、数学术语、双关语、数字和遗漏，但只产生
`translation_semantic_issue`和建议译文，绝不自动覆盖DeepL结果。需要开启时写入：

```json
"translation_verify": "always"
```

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

### 独立页级Codex最终复核（实验性）

`codex_page_review.py`只从`page-NNNN.json`的`study.regions`提取英文、当前译文
和同页区域上下文，并可按需把同名PNG交给本机Codex CLI直接裁决。它不会读取或发送
页面Markdown、`issues`、`translation_verifier_comparison`或`study.skipped`。
脚本复用`codex login`保存的ChatGPT登录，
会从子进程环境移除`OPENAI_API_KEY`和`CODEX_API_KEY`，不使用API Platform余额。
首次使用前需在终端运行`codex login`并选择ChatGPT登录。

模型和推理强度只在`ocr_config.json`中配置，例如：

```json
"codex_page_review": {
  "command": "codex",
  "model": "gpt-5.6-terra",
  "reasoning_effort": "high",
  "timeout_seconds": 1800,
  "image_mode": "on_demand",
  "require_chatgpt_login": true
}
```

`image_mode`支持：

- `on_demand`：默认。第一阶段只传文本；仅当某个区域明确需要图片证据时，
  第二阶段才附加PNG并只复核这些区域；
- `never`：始终不传图片；
- `always`：第一阶段直接附加整页图片。

先做不调用Codex的本地输入检查：

```powershell
python .\codex_page_review.py `
  --config .\ocr_config.json `
  --page-json .\runs\study-batch-01\pages\page-0053.json `
  --dry-run
```

确认后由用户运行真实复核：

```powershell
python .\codex_page_review.py `
  --config .\ocr_config.json `
  --page-json .\runs\study-batch-01\pages\page-0053.json
```

为兼容旧命令，`--page-md page-NNNN.md`仍可使用，但它只把文件名转换为同目录的
`page-NNNN.json`路径；Markdown本身不读取、不校验，也不进入输入哈希。

Codex在`read-only`沙箱中运行；脚本通过内部`reviewed_region_ids`验证
`study.regions`的全部区域都已检查。
最终文件的`decisions`只保存真正改变译文的`replace/normalize`项；当前译文可接受的
区域不会输出冗余决定。脚本在返回后重新核对实际使用的页面JSON和可选PNG的
SHA-256，验证成功后才原子写入
`page-NNNN-codex-review.json`。该实验性结果目前不会写回
页面JSON，也尚未由现有PDF回填入口自动读取；待独立验证通过后再集成最终译文快照。
结果中的`codex.usage`记录当次复核的`input_tokens`、`cached_input_tokens`、
`non_cached_input_tokens`、`output_tokens`、`reasoning_output_tokens`和`total_tokens`。
其中`reasoning_output_tokens`是输出Token的细分统计，不会再次加到`total_tokens`。
`codex.stages`分别记录`text`和可选`image`阶段的目标区域、耗时与Token；顶层
`codex.usage`是所有实际执行阶段的合计。

### 整书按目录拆分的Codex最终复核（实验性）

`codex_book_review.py`直接连接本机`codex app-server`，不依赖Python SDK或API Key。
目录解析规则和最终裁决规则通过`thread/start.developerInstructions`设置；用户消息只包含
目录页图片或当前批次页面的`study.regions`。程序只在同一个前置内容/正式章节内合并
连续页面，根据页面数、区域数和序列化Token软目标动态装箱；一个线程段通常承载4–6个
批次。新线程首批显式携带`translation.custom_instructions`、章节标题和上一批次保存的
精简一致性摘要，不会把跨章节页面放入同一批次或线程。

先让Codex解析用户指定的目录PDF页并生成任务计划：

```powershell
python .\codex_book_review.py plan `
  --config .\ocr_config.json `
  --pdf ".\book\beast academy math guide 3A.pdf" `
  --toc-pages 5-6 `
  --output .\runs\book-review-3A
```

计划文件为`book-review-plan.json`。程序核对PDF SHA-256、目录图片、目录条目顺序、
印刷页码到PDF页码偏移和任务范围。封面、版权/出版信息和目录页默认组成独立
`preliminary`翻译任务；只有传`--exclude-preliminary`或在配置中显式启用时才排除。
`Index`、`Appendix`、`Answers`和`Glossary`仍默认进入`excluded_ranges`；如确需包含，
可传`--include-back-matter`重新生成计划。

当计划覆盖范围内的`page-NNNN.json`已经由主OCR/翻译流程生成后，先做本地检查：

```powershell
python .\codex_book_review.py review `
  --config .\ocr_config.json `
  --pdf ".\book\beast academy math guide 3A.pdf" `
  --pages-dir .\runs\study-batch-3A\pages `
  --output .\runs\book-review-3A `
  --dry-run
```

去掉`--dry-run`后开始逐章复核。脚本先生成带配置快照和输入哈希的
`book-review-batch-plan.json`并锁定批次边界；配置或输入变化时必须显式使用`--force`
重建。脚本自身升级批次响应协议时，会保留已完成页并自动重建未完成批次，且使用新的
线程段，避免恢复旧协议的线程。`book-review-checkpoint.json`保存批次到线程段的
映射、thread ID、一致性摘要和已完成页，重启后自动`thread/resume`；批次审计结果写到
`batches/<task-id>/`，验证完整批次后再把每页独立结果写到
`chapters/<task-id>/page-NNNN-codex-review.json`，全书汇总写到
`book-codex-review-summary.json`。原页面JSON、`manifest.json`和`study.html`不会修改。

`codex_book_review`配置会继承`codex_page_review`中未重复指定的模型资料；默认仍为
`gpt-5.6-terra`、`high`和`on_demand`。每轮输出同时记录`input_tokens`、
`cached_input_tokens`、`cache_write_input_tokens`、`non_cached_input_tokens`、输出和总Token。
`codex_book_review.batching`中的页面、区域和新增Token值都是允许加入完整页面后略超的
软目标；容量硬限制按“预计输入 + 生成预留不超过模型上下文窗口的指定比例”计算。
更换模型时必须同步更新`model_context_window_tokens`。同一批次各逐页结果引用共享的
批次usage，全书汇总按`batch_id`去重，避免重复累计。
为避免模型回显时改写证据文本，批次响应的裁决以`region id`为键，只提交决定和最终译文；
程序从不可变的页面输入补回当前译文后再做严格验证。

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

新的整书流程不再依赖`ocr_demo.py`，并把最终合并、英文擦除和中文写入完全拆开。

先把全部章节Codex裁决合并为旧回写入口兼容的锁定快照和坐标计划：

```powershell
python .\codex_book_finalize.py `
  --config .\ocr_config.json `
  --book-review-dir .\runs\book-review-3A `
  --dry-run

python .\codex_book_finalize.py `
  --config .\ocr_config.json `
  --book-review-dir .\runs\book-review-3A
```

该脚本不调用模型。它验证书本计划、PDF和页面JSON SHA-256、全部`page + region id`、
稀疏裁决的当前译文、每章持久thread一致性及未解决人工项，然后生成：

- `book-codex-adjudication.json`：整书裁决聚合审计；
- `chapter_translation_final.json`：与旧回写方法格式一致的锁定最终译文快照；
- `pdf_backfill_plan.json`：最终译文到DeepSeek-OCR坐标的确定性映射。

英文擦除只由`erase_english_from_deepseek.py`完成。所有需翻译页生成
`page-NNNN.cleaned.png`后，纯回写器只把这些cleaned PNG作为页面底图并写入中文：

```powershell
python .\pdf_translation_writer.py `
  --config .\ocr_config.json `
  --cleaned-pages-dir .\runs\study-batch-3A\erase `
  --dry-run

python .\pdf_translation_writer.py `
  --config .\ocr_config.json `
  --cleaned-pages-dir .\runs\study-batch-3A\erase
```

`pdf_translation_writer.py`不包含mask、redaction、inpaint或其他擦除实现，也不导入
`erase_english_from_deepseek.py`。它要求每个快照页恰好存在一张cleaned PNG；未翻译的
后置页直接保留原PDF页面。输出保持原PDF完整页数，程序报告明确记录
`writer_performed_erasure=false`。程序检查通过仍只表示`program_checked`，最终视觉
接受必须由用户确认。

以下`ocr_demo.py --build-final-translation/--export-pdf`属于旧兼容流程，计划随
`ocr_demo.py`一起废弃，不应用于新的整书任务。

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
它使用原PDF页面作为底图，对最终确定写入中文的DeepSeek-OCR坐标生成
字形级掩码，以OpenCV局部修复英文，再透明叠加修复像素并写入矢量中文。
未进入中文写入流程的OCR区域不会被擦除。每个OCR坐标只能归属一条译文；
存在多个候选气泡时，只使用
匹配度最高的单个空间连通区，宁可保留少量英文，也不跨气泡覆盖。中文字号
按每个文本框独立自适应：先用类型默认字号，文字过多时逐级缩小，短文不放大；
默认最多缩小1点，保留轻微越界以避免中文过小。
宽度最多扩张到原框的1.15倍。允许少量越界和DeepSeek未定位文字的残留；找不到
可靠坐标的区域保留原文并写入报告，不阻断整章。中文使用独立STHeiti TrueType
字体嵌入，避免苹方TTC在macOS快速查看/PDFKit中出现乱码。
同一页需要回填的英文行框共用一次字形掩码修复；背景采样色只用于自动选择
中文深浅颜色，不再作为默认矩形填充色。局部覆盖文件显式指定
`background_color`时，仍可对该区域使用纯色人工修正。
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

如果只需要逐页JSON，不需要逐页Markdown，请改用独立入口
`ocr_json_pipeline.py`。它仍会生成`manifest.json`、`summary.md`和学习模式的
`study.html`，并保留OCR、OCR复核与DeepL翻译流程：

```powershell
python .\ocr_json_pipeline.py `
  --pdf ".\book\beast academy math guide 3A.pdf" `
  --pages "1-100" `
  --output ".\runs\study-batch-3A" `
  --mode study `
  --workers 2 `
  --verify auto
```

使用`--resume`续跑时，该入口只要求已有的`page-NNNN.json`，不会要求被省略的
`page-NNNN.md`。如需每页都调用OCR复核模型，将`--verify auto`改成
`--verify always`。

## 测试

```bash
python3 -m unittest discover -s tests -v
```

## 已验证批次

`runs/study-batch-03`覆盖24个代表页面和359个翻译条目。页面类型包括目录、
索引、普通/黑底漫画、手写笔记、百数表、定义页和几何图。详细评估见
`runs/study-batch-03/evaluation.md`。




### 删除DeepL授权

关闭正在运行的批处理后，删除`.env`中的整行
`DEEPL_OAUTH_CREDENTIALS=...`。下次翻译时程序会重新打开DeepL授权页。
