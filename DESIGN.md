# 系统设计

## 范围

系统只保留五段生产流程：

1. `ocr_json_pipeline.py`完成页面渲染、版面OCR、主OCR、可选OCR复核和DeepL翻译；
2. `codex_book_review.py`按目录拆分整书并执行跨页翻译复核；
3. `codex_book_finalize.py`聚合裁决，生成锁定译文快照和坐标计划；
4. `erase_english_from_deepseek.py`按页或按目录生成已擦除英文的PNG；
5. `pdf_translation_writer.py`把锁定译文写入cleaned PNG并组装完整PDF。

## 数据流

```text
源PDF
  -> Poppler页面PNG
  -> DeepSeek版面坐标 + Qwen主OCR
  -> 风险触发的独立OCR复核
  -> DeepL逐区域翻译
  -> page-NNNN.json + page-NNNN.png
  -> Codex分章持久复核
  -> page-NNNN-codex-review.json
  -> 最终裁决聚合
  -> chapter_translation_final.json + pdf_backfill_plan.json

page-NNNN.png + page-NNNN.json
  -> 英文擦除
  -> page-NNNN.cleaned.png

源PDF + 锁定译文 + 坐标计划 + cleaned PNG
  -> 中文写入器
  -> 中文PDF + 程序检查报告
  -> 人工视觉验收
```

## 责任边界

- 主OCR是英文权威来源；坐标OCR和OCR复核只产生证据，不自动覆盖主OCR。
- DeepL只翻译主OCR选出的学习区域，数字、公式和条件必须保持不变。
- Codex复核读取原英文、当前译文、上下文、术语和必要图像后直接裁决，不使用多数票。
- Codex结果不得写回页面JSON、`manifest.json`或`study.html`。
- 定稿前必须验证PDF、页面JSON和复核输入SHA-256，以及全部`page + region id`。
- 人工复核只处理Codex补充证据后仍不能决定的项目；清单可以为空。
- PDF回写只能读取锁定的最终译文快照，不能自行改译文。
- 擦除器负责mask与inpaint；中文写入器只消费cleaned PNG，不包含擦除实现。
- 程序检查通过只表示`program_checked`，最终视觉接受由用户确认。

## 配置边界

- API密钥与DeepL OAuth凭据只位于`.env`；
- `model_profiles`定义平台、模型ID、端点和思考开关；
- `model_usage`只把版面OCR、主OCR和OCR复核用途映射到模型资料；
- Qwen思考开关使用`enable_thinking`，DeepSeek使用`thinking.type`；
- `codex_page_review`和`codex_book_review`使用本机ChatGPT登录；
- `pdf_backfill`只定义快照、坐标计划和人工覆盖位置；
- `pdf_translation_writer`只定义cleaned PNG、输出、字体和排版规则。

## 不可变输入与独立输出

| 输入/产物 | 写入者 | 后续约束 |
|---|---|---|
| `page-NNNN.json` | OCR主链 | 后续流程只读 |
| `page-NNNN.png` | OCR主链 | 擦除与必要图像复核只读 |
| `book-review-plan.json` | 整书计划 | 定稿时校验 |
| `page-NNNN-codex-review.json` | Codex复核 | 不写回页面JSON |
| `book-codex-adjudication.json` | 定稿器 | 独立审计记录 |
| `chapter_translation_final.json` | 定稿器 | PDF回写唯一译文源 |
| `pdf_backfill_plan.json` | 定稿器 | 译文到坐标的确定性映射 |
| `page-NNNN.cleaned.png` | 擦除器 | 写入器的页面底图 |
| 中文PDF | 写入器 | 进入人工视觉验收 |

## 验证层级

1. Python语法、单元测试和JSONC模板解析；
2. 各脚本`--dry-run`的路径、哈希、区域集合与依赖检查；
3. 用户运行真实OCR、DeepL和Codex调用；
4. PDF结构、计数、字体、坐标唯一性、相交与非空检查；
5. 用户完成最终视觉验收。
