# music-insight

把音频转成 **LLM 可读的证据包**：客观测量 + 图 + 事实性报告。

文本模型读不到音频。这个项目不训练新模型，而是回到最朴素的思路：
**把音乐变成模型能读的东西**——数字、时间线、频谱图，以及一份每条结论都能追溯来源的报告。

```
音频 ──▶ 测量层（确定性算法）──▶ 证据层（JSON/CSV/PNG/Markdown）──▶ 任意 LLM 解读
```

纯 CPU、零 API、离线可跑（可选依赖用于解码与非 WAV 格式）。

![测量面板示例](docs/panel-example.png)

## 现在能做什么（实测数字）

在 [POP909](https://github.com/music-x-lab/POP909-Dataset)（909 首中文流行）的**合成钢琴音频**上评估：

| 能力 | 水平 | 说明 |
|---|---|---|
| 时长 / BPM / 响度 / 明暗 / 结构 | ✅ 可用 | BPM 误差通常在 ±2% 内 |
| 和弦（根音 / 性质） | 🟡 **73.8% / 73.0%** | 用 librosa 拍点；真实录音更低 |
| 调性 | 🟡 关系大小调二选一 | 由和弦序列反推，歧义会明确标注 |
| 旋律 | 🔴 不可用（19.5%） | 复调遮蔽，见「已知限制」 |

> 以上是**合成音频**上的数字，真实录音会更差。完整评估过程与失败记录见 [docs/EVAL.md](docs/EVAL.md)。

## 它是什么 / 不是什么

| 是 | 不是 |
|---|---|
| 把音频变成可复现的测量数据 | 音乐理解模型 |
| 生成给 LLM 读的证据包和提示词 | 自动乐评 SaaS |
| 明确标注置信度与失效条件 | 情绪识别 / 扒谱工具 |

## 快速开始

```bash
git clone <repo>
cd music-insight
pip install -e ".[decode]"        # numpy + pillow，av 用于 mp3/m4a/flac
pip install -e ".[dev]"           # 跑测试

# 1. 生成一段无版权的测试音频（100 BPM，I–V–vi–IV，19.2 秒）
python examples/make_demo.py out/demo.wav

# 2. 基础分析：测量 + 图 + 报告
music-insight out/demo.wav --out out/demo

# 3. 加上节拍 / 和弦 / 由和弦反推调性
music-insight out/demo.wav --out out/demo --chords

# 4. 批量分析一个文件夹
python examples/batch.py "某处/音乐文件夹" --out out/batch
```

输出：

```
out/demo/
├── panel.png      # 波形 + 梅尔谱 + 音高类 + 结构条
├── analysis.json  # 全部测量值、结构、情绪先验、证据索引
├── timeline.csv   # 8 秒粒度时间线
└── report.md      # 事实性报告（每条结论指向字段）
```

## 输出长什么样

`--chords` 会多出第 4 节（完整示例见 [docs/example-report.md](docs/example-report.md)）：

```markdown
## 4. 和弦（节拍同步）

- ⚠️ 关系大小调二选一：**F 大调** / D 小调
  两者共享全部调内和弦，仅凭和弦序列无法区分，需要人耳或旋律确认。
- 高频和弦：F 69.6s / Dm 43.2s / C 21.6s

| 起止（秒） | 和弦 |
|---|---|
| 12.4 – 14.8 | F |
| 14.8 – 17.1 | Dm |
```

## 设计原则：测量与解读分开

- **测量层**只放确定性算法，每个数字都能复现、能被基准检验。
- **情绪先验**单独成字段（`mood_priors`），显式标注 `method: heuristic-v0`、`confidence: low`，
  并写明失效条件。它不是测量值。
- **报告每条结论**都在「证据索引」里指向具体字段，幻觉率可以自动统计。

## 已知限制

- **旋律提取不可用**：完整混音上 RPA 19.5%，单独旋律轨 68–77%——瓶颈是复调遮蔽。
  试过 demucs，已否决（它只有 drums/bass/vocals/other，拆不出旋律）。
- **和弦性质依赖语料先验**：`QUALITY_PRIORS` 来自中文流行，换到古典/爵士会偏。
- **调性只能到关系大小调层级**：两者共享全部调内和弦，报告会并列两个候选。
- **节拍跟踪较弱**：自研 F-measure 0.21，librosa 0.34；对和弦聚合影响 5–6 个点。
- **和弦精度随音源下降**：数字来自合成钢琴音频，真实录音未验证。
- **解码器覆盖不全**：AVS3 音频（`av3a`）当前 FFmpeg 构建不支持，需先转 mp3/flac。
- **无流派 / 年代 / 乐器识别**：需要另外的嵌入模型。
- 纯音乐上歌词转写会产生幻觉，`--lyrics` 默认关闭且必须检查 `no_speech_prob`。

## 测试

```bash
pip install -e ".[dev]"
pytest -q      # 12 个测试，不需要下载数据集
```

## Roadmap

1. 旋律提取（dominant melody extraction，例如 Basic Pitch）——当前最大空白
2. 和弦继续提升：更多和弦类型、更大评测集、真实录音验证
3. 调性定到绝对调名（需要旋律或终止式信息）
4. CLAP / MERT 嵌入 → 流派与情绪标签（有监督、可评估）
5. 作为插件接入 DeepSeek Harness 等 agent 框架

## Prior art

- [MOSS-Music-Data-Pipeline](https://github.com/wx9songs/MOSS-Music-Data-Pipeline) — 训练语料标注管线
- [review-music](https://lobehub.com/skills/grahama1970-agent-skills-review-music) — MIR → 结构化评论，思路最接近
- [Musia](https://github.com/lachlanchen/Musia) — 本地音乐工具链封装
- LTU / Music Flamingo — 训练音频编码器的路线，需要 GPU 与数据

差异点：**证据与解读强制分离**、**零 API 纯本地模型无关**。

## 许可

MIT
