# Beast Academy OCR 与中文 PDF 流程

项目使用统一入口完成五个阶段：

```text
OCR、OCR复核、DeepL翻译
  -> Codex整书翻译复核
  -> 最终裁决和锁定快照
  -> 擦除英文
  -> 写入中文PDF
```

## 目录

```text
ocr-demo/
├── pipeline/                 # Python包和五阶段实现
├── tools/                    # DeepL MCP辅助程序
├── config/
│   ├── pipeline.example.json
│   ├── pipeline.json         # 本地运行配置，不提交
│   ├── .env.example
│   ├── .env                  # 密钥与OAuth凭据，不提交
│   └── overrides/            # 可选的每书人工覆盖文件
├── docs/
├── tests/
├── book/                     # 输入PDF
└── runs/book/                # 自动生成的分类产物
```

## 环境

- Python 3.11或更高版本；
- Poppler的`pdftoppm`和`pdfinfo`；
- Node.js 18或更高版本；
- 已使用ChatGPT帐号登录的Codex CLI。
- 将可用的中文 TrueType/OpenType 字体放入`assets/fonts/simhei.ttf`（该路径由配置模板使用）。

项目优先使用`.conda`环境：

```powershell
.\.conda\python.exe -m pip install -r .\requirements.txt
npm install
codex login status
```

复制配置模板，并只在`config/.env`填写密钥：

```powershell
Copy-Item .\config\pipeline.example.json .\config\pipeline.json
Copy-Item .\config\.env.example .\config\.env
```

## 输入和自动路径

单本书：

```jsonc
"pdf": "./book/beast academy math guide 3A.pdf"
```

目录批处理：

```jsonc
"pdf": "./book/"
```

相对路径以项目根目录为基准。目录模式按文件名排序处理直属的全部`.pdf`，不递归
子目录。PDF文件名去扩展名后，空白和不适合作为目录名的字符转换为下划线。

例如`beast academy math guide 3A.pdf`自动使用：

```text
runs/book/beast_academy_math_guide_3A/
├── 01-ocr/
│   ├── info.log
│   ├── manifest.json
│   ├── summary.md
│   ├── study.html
│   └── pages/
├── 02-codex-review/
│   ├── book-review-plan.json
│   ├── book-review-checkpoint.json
│   ├── book-codex-review-summary.json
│   └── chapters/
├── 03-final/
│   ├── book-codex-adjudication.json
│   ├── chapter_translation_final.json
│   └── pdf_backfill_plan.json
├── 04-erased/
│   ├── page-NNNN.cleaned.png
│   ├── page-NNNN.mask.png
│   ├── page-NNNN.debug.png
│   └── page-NNNN.adjusted.json
└── 05-pdf/
    ├── beast_academy_math_guide_3A-zh-review.pdf
    └── pdf_translation_writer_report.json
```

这些路径不写入配置，也不需要在命令行手动指定。

## 五步命令

默认读取`config/pipeline.json`：

```powershell
.\.conda\python.exe -m pipeline ocr
.\.conda\python.exe -m pipeline review
.\.conda\python.exe -m pipeline finalize
.\.conda\python.exe -m pipeline erase
.\.conda\python.exe -m pipeline erase-v2       # 可选：用V2替代V1
.\.conda\python.exe -m pipeline write
```

指定其他配置时，`--config`放在阶段名称之前：

```powershell
.\.conda\python.exe -m pipeline --config .\config\pipeline.local.json ocr
```

### 1. OCR

`ocr`渲染配置页码，并行执行DeepSeek版面定位和Qwen主OCR。`verify: auto`只对
确定性风险页调用独立图像OCR复核。主OCR仍是英文权威来源，复核不会自动覆盖它。
学习区域随后由DeepL逐区域翻译。

本地预检查：

```powershell
.\.conda\python.exe -m pipeline ocr --dry-run
```

### 2. Codex复核

`review`在`02-codex-review`不存在计划时，先使用配置中的`toc_pages`生成计划，再按
章节持久任务复核`01-ocr/pages`。已有计划时直接恢复或继续复核。

```powershell
.\.conda\python.exe -m pipeline review --dry-run
```

真实Codex调用由用户去掉`--dry-run`后运行。

### 3. 定稿

`finalize`验证源PDF、页面JSON、计划、裁决、全部`page + region id`和输入SHA-256，
再把产物写入`03-final`。

```powershell
.\.conda\python.exe -m pipeline finalize --dry-run
```

### 4. 擦除英文

`erase`自动读取`01-ocr/pages`的全部同名PNG/JSON页面对并写入`04-erased`：

```powershell
.\.conda\python.exe -m pipeline erase --dilate-iterations 2
```

也可以使用V2擦除器；它采用相同的自动输入和输出路径，不需要传入目录参数：

```powershell
.\.conda\python.exe -m pipeline erase-v2
```

`python -m pipeline.erase_v2`无参数运行时等价于上述统一入口命令。

擦除器不写中文，也不修改页面JSON。

### 5. 写入中文PDF

`write`自动读取`03-final`的锁定快照和坐标计划，以及`04-erased`的cleaned PNG：

```powershell
.\.conda\python.exe -m pipeline write --dry-run
```

确认后去掉`--dry-run`。程序检查通过只能标记`program_checked`，最终视觉接受由用户
确认。

## 人工覆盖

若存在以下文件，统一入口会自动加载：

```text
config/overrides/<书名>.translations.json
config/overrides/<书名>.layout.json
```

译文覆盖只在定稿阶段读取；布局覆盖只在PDF写入阶段读取。两者均不修改页面JSON或
Codex原裁决。

## 本地验证

```powershell
.\.conda\python.exe -m py_compile `
  .\pipeline\__main__.py .\pipeline\config.py .\pipeline\paths.py `
  .\pipeline\ocr.py .\pipeline\page_review.py .\pipeline\codex_review.py `
  .\pipeline\finalize.py .\pipeline\erase.py `
  .\pipeline\pdf_backfill.py .\pipeline\pdf_writer.py
.\.conda\python.exe -m unittest discover -s tests
```

语法、单元测试和`--dry-run`属于本地验证，不等于真实外部API端到端验证。
