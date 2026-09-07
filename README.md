# Beast Academy OCR 与中文 PDF 流程

项目只保留以下生产链：

```text
OCR + OCR复核 + DeepL翻译
  -> Codex整书翻译复核
  -> 最终裁决快照与坐标计划
  -> 擦除英文
  -> 写入中文PDF
```

页面JSON、`manifest.json`和`study.html`是不可变证据；Codex复核与后处理只生成
独立产物。

## 环境

- Python 3.11或更高版本；
- Poppler的`pdftoppm`和`pdfinfo`；
- Node.js 18或更高版本及`npm install`安装的DeepL MCP依赖；
- `requirements.txt`中的PDF与图像处理依赖；
- 已使用ChatGPT帐号登录的Codex CLI。

项目优先使用已有`.conda`环境。API密钥和DeepL OAuth凭据只存放于被忽略的
`.env`，不要写进`ocr_config.json`。

## 1. OCR、OCR复核和DeepL翻译

```powershell
.\.conda\python.exe .\ocr_json_pipeline.py `
  --pdf ".\book\beast academy math guide 3A.pdf" `
  --pages "1-100" `
  --output ".\runs\study-batch-3A" `
  --mode study `
  --workers 2 `
  --verify auto
```

主要产物位于`runs\study-batch-3A\pages`：每页一张PNG和一个
`page-NNNN.json`。`--verify auto`只对风险页调用独立图像OCR复核；主OCR仍是
英文权威来源，复核结果不会自动覆盖主OCR。学习区域随后由DeepL逐区域翻译。

首次运行前执行`npm install`。DeepL首次调用会完成OAuth登录，并在`.env`中维护
`DEEPL_OAUTH_CREDENTIALS`。

## 2. Codex整书翻译复核

先用`plan`命令根据目录页生成`book-review-plan.json`；已有计划时可直接执行
`review`：

```powershell
.\.conda\python.exe .\codex_book_review.py review `
  --config .\ocr_config.json `
  --pdf ".\book\beast academy math guide 3A.pdf" `
  --pages-dir .\runs\study-batch-3A\pages `
  --output .\runs\book-review-3A
```

真实Codex调用由用户运行。建议先加`--dry-run`检查PDF、计划、页面输入和缺失项。
每个章节使用可恢复的持久任务，结果写到
`runs\book-review-3A\chapters\<task-id>`，不会修改页面JSON。

## 3. 生成最终裁决文件

```powershell
.\.conda\python.exe .\codex_book_finalize.py `
  --config .\ocr_config.json `
  --book-review-dir .\runs\book-review-3A
```

建议先加`--dry-run`。脚本校验输入SHA-256、全部`page + region id`、裁决和未解决
人工项，然后生成：

- `book-codex-adjudication.json`；
- `chapter_translation_final.json`；
- `pdf_backfill_plan.json`。

`chapter_translation_final.json`是回写阶段唯一可读取的最终译文快照。

## 4. 擦除英文

批量处理目录：

```powershell
.\.conda\python.exe .\erase_english_from_deepseek.py `
  --pages-dir .\runs\study-batch-3A\pages `
  --output-dir .\runs\study-batch-3A\erase `
  --dilate-iterations 2
```

单页调试：

```powershell
.\.conda\python.exe .\erase_english_from_deepseek.py `
  --image .\runs\study-batch-3A\pages\page-0030.png `
  --json .\runs\study-batch-3A\pages\page-0030.json `
  --output-dir .\runs\study-batch-3A\erase `
  --dilate-iterations 2
```

目录模式按文件名处理同名PNG/JSON页面对，生成`page-NNNN.cleaned.png`。

## 5. 回写中文PDF

```powershell
.\.conda\python.exe .\pdf_translation_writer.py `
  --config .\ocr_config.json `
  --cleaned-pages-dir .\runs\study-batch-3A\erase
```

建议先加`--dry-run`检查锁定快照、坐标计划、字体和cleaned PNG是否齐全。
写入器不执行擦除、不重新判断译文，也不修改快照。程序检查通过只能标记
`program_checked`；最终视觉接受由用户确认。

## 配置

`ocr_config.example.json`保留五步流程需要的内容：

- OCR、OCR复核、DeepL和模型用途映射；
- `codex_page_review`与`codex_book_review`；
- `pdf_backfill`与`pdf_translation_writer`。

配置支持JSONC注释。复制模板后只修改路径、模型资料和用途映射；密钥仍放`.env`。

## 本地验证

```powershell
.\.conda\python.exe -m py_compile `
  .\ocr_json_pipeline.py `
  .\codex_page_review.py `
  .\codex_book_review.py `
  .\codex_book_finalize.py `
  .\erase_english_from_deepseek.py `
  .\pdf_backfill.py `
  .\pdf_translation_writer.py
.\.conda\python.exe -m unittest discover -s tests
```

本地语法、单元测试和`--dry-run`不等于真实外部API端到端验证。
