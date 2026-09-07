# 系统设计

## 入口与模块

唯一公开入口是`python -m pipeline <stage>`。`pipeline/__main__.py`读取共享配置，
发现一个或多个PDF，为每本书生成`BookPaths`，再调用对应阶段：

| 阶段 | 模块 | 职责 |
|---|---|---|
| `ocr` | `pipeline.ocr` | 渲染、版面OCR、主OCR、OCR复核、DeepL翻译 |
| `review` | `pipeline.codex_review` | 目录计划与分章持久Codex复核 |
| `finalize` | `pipeline.finalize` | 裁决聚合、锁定译文、坐标计划 |
| `erase` | `pipeline.erase` | 字形掩码、OpenCV修复、cleaned PNG |
| `write` | `pipeline.pdf_writer` | 中文排版、PDF组装、程序检查 |

`pipeline.page_review`和`pipeline.pdf_backfill`是内部共用模块，不作为独立生产入口。
`pipeline.config`负责JSONC读取，`pipeline.paths`是所有派生路径的唯一来源。

## PDF发现与路径不变式

配置`pdf`可以是单个PDF或目录。相对路径以项目根目录为基准；目录只扫描直属PDF，
按文件名排序，不递归。每本PDF的文件名生成唯一书名slug。

所有运行产物固定归入：

```text
runs/book/<slug>/
  01-ocr/
  02-codex-review/
  03-final/
  04-erased/
  05-pdf/
```

阶段模块不自行拼接业务目录；统一入口显式把`BookPaths`中的绝对路径传给模块。
若同批PDF规范化后得到重复slug，必须在写入前停止。

## 数据流

```text
源PDF
  -> 01-ocr: 页面PNG、页面JSON、manifest、summary、study.html
  -> 02-codex-review: 计划、检查点、逐页独立裁决
  -> 03-final: 聚合裁决、锁定译文、坐标计划

01-ocr页面PNG/JSON
  -> 04-erased: cleaned PNG、mask、debug、adjusted JSON

源PDF + 03-final + 04-erased
  -> 05-pdf: 中文PDF、程序检查报告
  -> 人工视觉验收
```

## 配置边界

`config/pipeline.json`只保存不可自动推导的输入和行为：

- PDF文件或目录、页码、目录页、并发和复核策略；
- `model_profiles`与`model_usage`；
- DeepL连接和翻译约束；
- Codex模型、推理强度和图像策略；
- PDF字体与布局规则。

配置不得保存书籍输出根目录、日志、页面目录、复核目录、擦除目录或输出文件名。
API密钥与DeepL OAuth凭据只位于`config/.env`。

模型资料和用途映射保持解耦。Qwen思考开关使用`enable_thinking`；DeepSeek协议转换
仍由调用层负责。

## 证据与裁决不变式

- 主OCR是英文权威来源；坐标OCR和OCR复核只产生证据。
- 页面JSON、`manifest.json`和`study.html`在Codex复核后保持不变。
- Codex读取原英文、当前译文、同章上下文、术语和必要图片后直接裁决，不用多数票。
- 定稿前验证PDF、页面JSON、计划、裁决输入SHA-256和全部`page + region id`。
- 人工复核只处理Codex补充证据后仍无法判断的项目；清单可以为空。
- `chapter_translation_final.json`是PDF写入器唯一可读取的译文来源。
- 擦除与写入分离；写入器不包含mask、redaction或inpaint实现。
- 程序检查只可标记`program_checked`；最终视觉接受必须由用户确认。

## 独立覆盖文件

人工译文与布局修正按slug自动发现：

```text
config/overrides/<slug>.translations.json
config/overrides/<slug>.layout.json
```

人工覆盖不得写回OCR证据或Codex裁决。译文覆盖优先级高于Codex裁决；布局覆盖只能
改变坐标、字号、颜色等排版属性，不能改变译文。

## 验证层级

1. 包内模块语法、单元测试和配置解析；
2. 统一入口的PDF发现、slug及路径映射测试；
3. 各阶段`--dry-run`的路径、哈希、区域集合和依赖检查；
4. 用户运行真实OCR、DeepL与Codex；
5. PDF程序全量检查；
6. 用户完成最终视觉验收。
