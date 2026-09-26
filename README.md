# EchoLingo

简体中文 | [English](README.en.md)

EchoLingo 是一款面向课堂的实时同声字幕工具：把老师的讲课声音实时转成原文与译文逐句对照的双语字幕，下课后还可以用你自己选择的 AI 模型把转录整理成学习笔记。默认所有识别和翻译都在你自己的电脑上完成。支持 Mac、Windows 和 Linux。

[![Platform](https://img.shields.io/badge/platform-macOS%2014%2B%20%C2%B7%20Windows%20%C2%B7%20Linux-lightgrey)](#系统要求)
[![Version](https://img.shields.io/badge/version-0.3.1%20Beta-orange)](https://github.com/Asphr726/EchoLingo/releases)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

![EchoLingo 实时双语字幕](docs/screenshots/live.png)

> **v0.3.1 Beta**：支持 Apple Silicon Mac（macOS 14 及以上）、Windows 10 / 11（x64）和 Linux（Ubuntu 22.04+ / Debian 12+，x64），Windows 和 Linux 的安装方法见 [Windows / Linux](#windows--linux)。应用界面暂时只有英文，下文中的按钮和菜单名称都按界面上的英文原样写出。

[特色](#特色) · [系统要求](#系统要求) · [安装](#安装) · [Windows / Linux](#windows--linux) · [快速上手](#快速上手) · [使用指南](#使用指南) · [隐私](#隐私) · [常见问题](#常见问题) · [路线图](#路线图) · [从源码构建](#从源码构建) · [许可与致谢](#许可与致谢)

## 特色

- **本地优先，保护隐私**。语音识别（Qwen3-ASR）和翻译（Hy-MT2）默认都在本机运行，不需要账号，模型下载完成后可以离线使用。只有你主动启用某个云服务时，音频或文字才会离开这台电脑，而且每一种上传都要先经过你明确同意。API 密钥保存在系统的安全存储里（macOS 钥匙串、Windows 凭据管理器或 Linux 的 Secret Service）。
- **为课堂远距离收音设计**。默认的 Lecture / far-field 音频配置会先在本机做降噪和自动增益（AGC）。语音检测只用来做标记，不会丢掉任何声音。一句话要等停顿确认结束后才会定稿，尽量避免把一句话从中间切断。
- **双语对照字幕和悬浮字幕窗**。主窗口里左边是原文、右边是译文，逐句对应。悬浮字幕窗可以一直浮在幻灯片、PDF 或笔记软件上面，字号、透明度和显示内容都能调整。
- **课程上下文**。上课前填好本节课的主题、人名和术语（可以写成 `术语 = 译法`），也可以从课件一键导入。识别和翻译都会参考这些信息，专有名词和术语更准确。长期使用的术语放进全局术语表即可。
- **AI 笔记**。AI 笔记会把口语化、零散的课堂转录整理成结构清晰的学习笔记，用你设置的翻译语言书写，公式用 LaTeX 显示。整理时还可以附上课程资料（PDF、PPTX、DOCX、Markdown、LaTeX、纯文本）。笔记和这节课一起保存，每一小节都标有时间段，点一下就能跳回对应的原文。笔记由你自己选择的聊天模型来写：可以是使用你自己密钥的云服务，也可以是本机运行的 Ollama 等。
- **自动命名**。设置好 AI 助手后，下课点 **Stop** 时它会自动为这节课起一个标题。你自己改过的标题不会被覆盖。
- **可插拔的云服务**。需要时可以改用云端识别或翻译，两者可以分开选择，例如本地识别配合云端翻译。
  - 识别：Qwen Cloud（阿里云百炼）、OpenAI Realtime、Deepgram、AssemblyAI（仅英语）、Gladia。
  - 翻译：Qwen-MT、兼容 OpenAI 接口的聊天模型（OpenAI、DeepSeek、Google Gemini、Groq、OpenRouter、硅基流动，以及 Ollama、LM Studio、vLLM 等自定义端点）、DeepL、Google Cloud Translation、Azure AI Translator。
  - 阿里云百炼的北京地域为新账号提供免费额度（每个模型 90 天）。
- **可搜索的历史记录**。按标题或转录内容搜索，可以重命名、删除，也可以导出为 Markdown、TXT、JSON、SRT、VTT 字幕，或者单独导出 AI 笔记。
- **支持语言**：英语、中文、日语、韩语，其中任意两种之间可以互译。

## 系统要求

| 项目 | macOS | Windows / Linux |
| --- | --- | --- |
| 系统 | macOS 14 Sonoma 或更新版本 | Windows 10 21H2 及以上或 Windows 11（x64）；Ubuntu 22.04 及以上或 Debian 12 及以上（x64） |
| 芯片 | Apple Silicon（M1 或更新），不支持 Intel Mac | 支持 AVX2 的 x64 处理器；可选 NVIDIA 显卡，用于 GPU 加速包 |
| 内存 | 建议 16 GB 或以上 | 建议 16 GB 或以上 |
| 磁盘 | 本地模型共约 4.5 GB，建议预留 6 GB 以上的可用空间 | 同左；可选的 GPU 加速包另需约 6 GB（Windows）或 8.5 GB（Linux） |
| 权限 | 麦克风；如果要采集电脑播放的声音（网课、视频），还需要“屏幕与系统录音”权限 | Windows：需要允许桌面应用使用麦克风；Linux：无需授权 |
| 网络 | 只有下载模型、检查更新、使用云服务或 AI 笔记时才需要 | 同左 |

在 Windows 和 Linux 上，没有安装 GPU 加速包时识别和翻译都在 CPU 上运行。处理器较慢时字幕可能会越来越落后于老师的讲课，这时可以安装 [GPU 加速包](#windows--linux)，或者改用云端识别服务。

## 安装

以下是 macOS 的安装步骤。Windows 和 Linux 请看 [Windows / Linux](#windows--linux)。

1. 从 [GitHub Releases](https://github.com/Asphr726/EchoLingo/releases) 下载 `EchoLingo_0.3.1_aarch64.dmg`。
2. 打开 dmg，把 **EchoLingo** 拖进“应用程序”文件夹。
3. 第一次打开。Beta 版还没有经过 Apple 公证，macOS 会拦截它：
   - **macOS 14**：在“应用程序”里按住 Control 点按（或右键点按）EchoLingo，选择“打开”，再在弹出的对话框里点“打开”。
   - **macOS 15 及以上**：先双击打开一次，看到提示后点“完成”。然后打开“系统设置 → 隐私与安全性”，拉到页面底部，点 EchoLingo 旁边的“仍要打开”，输入密码确认。
   - 如果仍然提示“已损坏”或无法打开，可以在“终端”里运行下面这条命令，然后重新打开：

     ```bash
     xattr -dr com.apple.quarantine /Applications/EchoLingo.app
     ```

4. 下载本地模型。完成首次设置向导后，打开 **Settings → Models**，在 **Local model manager** 里点每个模型右侧的 **Download**：

   | 模型 | 用途 | 大小 |
   | --- | --- | --- |
   | Qwen3-ASR 0.6B | 语音识别 | 1.7 GB |
   | Hy-MT2 1.8B（Q4_K_M 量化） | 翻译 | 1.1 GB |
   | Qwen3 ForcedAligner 0.6B | 下课后把时间戳校准到逐字精度（可选，推荐） | 1.7 GB |

   每一行都会显示下载进度，下载完成后会自动校验文件完整性。已下载的模型可以随时点 **Verify** 重新校验，或者点 **Delete** 删除。如果下载失败或被中断，再点一次 **Download** 即可；如果已下载的模型文件不完整，这一行会显示 **Retry**，点它重新下载。模型从 Hugging Face 下载，只需要下载一次。

   > **中国大陆用户**：如果下载很慢或者一直失败，可以试试 Hugging Face 镜像。在“终端”里运行
   > `launchctl setenv HF_ENDPOINT https://hf-mirror.com`，然后用 ⌘Q 完全退出 EchoLingo，重新打开后再点 **Download**。
   > 这个设置在注销或重启 Mac 后失效。想换回官方源，运行 `launchctl unsetenv HF_ENDPOINT` 后重新打开应用。

## Windows / Linux

除这里说明的差别外，应用和设置的用法与本文其他部分相同。Windows 版已经在真实电脑上测试过；Linux 版目前只在自动构建环境中测试过，还没有在真实电脑上测试。

**Windows 10 / 11（x64）**

1. 从 [GitHub Releases](https://github.com/Asphr726/EchoLingo/releases) 下载并运行 `EchoLingo_0.3.1_x64-setup.exe`。安装程序只为当前用户安装，不需要管理员权限；如果电脑上缺少 Microsoft Edge WebView2 运行时，安装程序会自动安装。
2. 安装程序还没有代码签名，Windows SmartScreen 可能提示“Windows 已保护你的电脑”。点“更多信息”，再点“仍要运行”。
3. 从“开始”菜单打开 EchoLingo，按照[安装](#安装)第 4 步在 **Settings → Models** 下载本地模型。

**Linux（Ubuntu 22.04+ / Debian 12+，x64）**

1. 从 [GitHub Releases](https://github.com/Asphr726/EchoLingo/releases) 下载 `EchoLingo_0.3.1_amd64.deb`，在下载目录里运行 `sudo apt install ./EchoLingo_0.3.1_amd64.deb` 安装。目前只提供 .deb 包，AppImage 在路线图中。
2. 从应用菜单打开 EchoLingo，或在终端运行 `echolingo-desktop`，同样在 **Settings → Models** 下载本地模型。

**NVIDIA GPU 加速包（可选）**。不安装时，识别和翻译都在 CPU 上运行，较慢的处理器可能跟不上讲课：在一台 i7-13700H 笔记本上，Qwen3-ASR 用 CPU 处理一段 11 秒的录音需要 2.07 倍的时长。同一台电脑用加速包（RTX 4060 Laptop）运行时，实时识别的延迟中位数为 0.1–0.2 秒、最长约 2 秒，Hy-MT 在 Vulkan 上每秒翻译约 135 个 token。

- **要求**：NVIDIA 显卡，计算能力 7.5 及以上（GeForce GTX 16 系列、RTX 20 系列或更新），驱动版本 580.65 或更新。AMD 和 Intel 显卡暂不支持。Linux 上用显卡翻译还需要 Vulkan 运行库（`sudo apt install libvulkan1`），没有它时翻译仍在 CPU 上运行。
- **磁盘空间**：Windows 上下载约 2.0 GB、解压后约 3.6 GB；Linux 上下载约 2.6 GB、解压后约 5.4 GB。安装时两者同时占用空间（装好后下载的文件会删除），都要算在本地模型之外。
- **安装**：打开 **Settings → Models → GPU acceleration**，点 **Download**。加速包分段下载，每一段都会校验，安装后先做一次自检再启用；之后可以随时用 **Use GPU acceleration** 开关关闭。如果 GPU 运行时启动失败，EchoLingo 会自动退回 CPU，并在这张卡片上说明原因。
- **每次更新应用后都要重新下载**：加速包和应用版本一一对应。更新 EchoLingo 后，在同一张卡片上重新下载即可；在此之前识别和翻译在 CPU 上运行。

**与 Mac 版的差别**

- **系统声音**：Windows 上选 **System audio** 会直接采集默认输出设备（扬声器或耳机）正在播放的声音，不需要选择窗口。Linux 暂不支持采集系统声音，请使用麦克风。
- **密钥**保存在 Windows 凭据管理器，或 Linux 上的 Secret Service（例如 GNOME 钥匙环或 KWallet）。如果都无法访问，**Settings → Cloud providers** 会给出提示，这时仍可以用环境变量（例如 `DEEPL_API_KEY`）提供密钥。
- **Hugging Face 镜像**：Windows 在“命令提示符”或 PowerShell 里运行 `setx HF_ENDPOINT https://hf-mirror.com`，然后退出并重新打开 EchoLingo（运行 `reg delete HKCU\Environment /v HF_ENDPOINT /f` 换回官方源）。Linux 在 `~/.profile` 里加一行 `export HF_ENDPOINT=https://hf-mirror.com` 并重新登录，或者在设置了这个变量的终端里启动 EchoLingo。
- **悬浮字幕**：透明度设置只在 macOS 上提供。

## 快速上手

1. **首次设置向导**。第一次打开时，向导会带你完成 7 步：欢迎、选择语言、选择音频来源和音频配置、选择推理模式、授予权限、测试音频、完成。每一项之后都可以在设置里修改。
2. **下载模型**。按照上面[安装](#安装)第 4 步，在 **Settings → Models** 下载模型。
3. **开始上课**。回到 **Live** 页面，确认源语言（老师讲的语言）和目标语言（你想看的语言），点 **Start**。第一次加载模型需要 1 到 2 分钟。之后 EchoLingo 每次启动都会在后台预加载模型，点 Start 就能立即开始。
4. **打开悬浮字幕**。点 **Floating caption**，把小字幕窗拖到幻灯片旁边。
5. **下课**。点 **Stop**，这节课会自动保存到 **History**。如果设置了 AI 助手并开启了自动命名，它会马上得到一个标题。

![首次设置向导](docs/screenshots/onboarding.png)

## 使用指南

### Live：实时字幕

![课程上下文面板](docs/screenshots/lecture-context.png)

- **Source language / Target language**：源语言和目标语言，可以在英、中、日、韩之间任选两种。
- **Audio source**：选 **Microphone**（麦克风，适合教室现场）或 **System audio**（电脑正在播放的声音，适合网课、录播视频、线上会议）。选择 System audio 后，macOS 每次开始时都会让你选择要采集的屏幕或 App；Windows 会直接采集默认输出设备；Linux 暂不提供 System audio。
- **Audio profile**：音频配置。课堂请保持 **Lecture / far-field**（默认，针对远距离收音）；面对面近距离交谈可以选 **Conversation**；**Raw diagnostics** 不做任何音频处理，只用于排查问题。
- **Input device**：选择使用哪个麦克风。
- **Inference mode**：推理模式。
  - **Local**：只用本机模型。
  - **Cloud**：使用云服务。
  - **Auto**：默认使用本机模型。点 Start 时，如果本机模型在这台电脑上跟不上实时速度，并且你已经允许了相应的上传，才会改用 **Settings → Cloud providers** 里设置的首选云服务。Auto 自己从来不会替你打开上传许可。
- 右侧的 **Inference** 面板可以分别指定识别服务（ASR provider）和翻译服务（Translation provider）。**Audio / VAD** 面板显示输入音量和是否检测到语音，**Latency** 面板显示各环节的延迟。最下面的卡片说明本节课的音频和文字会不会离开这台电脑；在 Auto 模式或选了云服务时，这里还有 **Allow audio upload** 和 **Allow transcript upload** 两个开关。
- **Start / Pause / Resume / Stop**：开始、暂停、继续、结束。
- 字幕区左边是原文，右边是译文。最下面标着 **LIVE** 的一行是正在识别、尚未定稿的句子，颜色较浅的部分还可能变化。一句话定稿后会固定下来，译文随后补上。往上翻看时会暂停自动滚动，新句子到来时也会停在你正在看的位置；点 **Back to live** 回到最新位置。

**悬浮字幕**：点 **Floating caption**（或在 **Settings → General** 点 **Open caption**）会打开一个始终置顶的小窗口，可以拖动和调整大小，显示最近 1 到 3 句以及正在说的那一句。在 **Settings → Appearance** 里可以选择显示原文和译文、只显示原文或只显示译文，还能调整字号（18 到 72 px）、透明度（仅 macOS）和显示的句数。

![悬浮字幕窗](docs/screenshots/caption.png)

**课程上下文（Lecture context）**：上课前展开这个面板，每行写一条：课程主题、人名、术语。写成 `pre-attentive vision = 前注意视觉` 这样的格式时，会作为本节课的术语对照。上限 2,000 个字符。

- 点 **Import from slides…** 可以从课件（PDF、PPTX、DOCX 等）中提取术语。没有设置 AI 助手时，文字只在本机提取。设置了 AI 助手但还没允许发送时，EchoLingo 会先询问能否把课件文字发给它来挑选术语和译法，选择 **Extract on this computer** 就只在本机提取；已经允许发送时，课件文字会直接发给 AI 助手，按钮旁边的提示会写明这一点。
- 课程进行中，上下文变为只读。
- 每节课都要用的术语请放到 **Settings → Translation → Glossary**（全局术语表），每行一条，`#` 开头的行是注释，上限 4,000 个字符，从下一节课开始生效。
- 使用本机模型时，上下文只在本机使用。使用云服务时，只有打开了对应的上传开关，上下文才会随音频或文字一起发给该服务。

### History：历史记录和 AI 笔记

![历史记录中的 AI 笔记](docs/screenshots/history-notes.png)

- **搜索和整理**：左侧搜索框可以搜索课程标题和转录内容。标题前带小星标（✦）的是 AI 起的标题。右上角的铅笔按钮用来重命名，垃圾桶按钮用来删除（会二次确认，并同时删除这节课的 AI 笔记）。仍是默认标题（例如 `en → zh lecture`）的课程会多一个星标按钮（**Generate title**），可以手动生成标题。
- **Transcript 标签**：逐句双语对照，带时间戳。**Export** 一栏可以导出 MD、TXT、JSON、SRT、VTT；如果这节课已经有 AI 笔记，导出的 Markdown 会把笔记放在最前面。下课后，本机的对齐模型会在后台把时间戳校准到逐字精度，此时 **Timing** 显示为 `word-aligned`。
- **AI notes 标签**：
  1. 先在 **Settings → AI assistant** 设置好 AI 助手（见下文）。
  2. 点 **Create notes**。EchoLingo 会询问要不要附上课程资料（**Add files…**），也可以直接点 **Create without files**。最多 5 个文件，每个不超过 25 MB，支持 PDF、PowerPoint（.pptx）、Word（.docx）、Markdown、LaTeX、CSV 和纯文本。文字在本机提取，只发送提取出的文字；没有文字层的扫描版 PDF 会被提示并跳过。
  3. 笔记一边生成一边显示，用这节课的翻译语言书写，包含标题、概述、分节内容、公式和关键术语。超过约 20 分钟的课程会按每段 12 到 15 分钟分批整理，最后合成一份完整的笔记。
  4. 每个小节标题旁边的时间段（如 `08:40–21:15`）可以点击，点击后跳到转录里对应的位置，再点 **Back to notes** 返回。左侧的 **Contents** 目录可以快速定位小节。
  5. 工具栏上有 **Copy**（复制 Markdown）、**Export .md** 和 **Regenerate**（新笔记生成完成后才会替换旧笔记）。笔记和课程一起保存，下次打开时直接显示，不会再次调用模型。

### Settings：设置

| 分区 | 内容 |
| --- | --- |
| General | 打开悬浮字幕；应用内更新（Updates） |
| Audio | 默认音频配置 |
| Languages | 默认源语言和目标语言 |
| Inference | 默认推理模式（Auto / Local / Cloud）；启动时是否预加载本地模型 |
| Models | 默认识别和翻译服务；下载、校验、删除本地模型；GPU 加速包（Windows 和 Linux） |
| Cloud providers | Auto 模式下的首选云端识别和翻译服务；每个服务商的密钥卡片 |
| AI assistant | AI 笔记和标题使用的模型、发送同意、自动命名 |
| Privacy | 两个上传开关：音频上传（Audio upload）和文字上传（Transcript upload） |
| Translation | 全局术语表 |
| Appearance | 悬浮字幕的外观 |
| Advanced | 诊断信息和日志目录 |

**Cloud providers（云服务）**：每个服务商一张卡片。粘贴密钥后点 **Save**，密钥会存进系统的安全存储（macOS 钥匙串、Windows 凭据管理器或 Linux 的 Secret Service）；点 **Clear** 可以删除它；点 **Test** 检查配置是否正确。

- 识别服务的测试只完成一次身份验证握手，不上传任何音频。
- 翻译服务的测试只发送一句固定的英文 “Welcome to the lecture.”，而且需要允许文字上传。还没允许时，只做翻译的服务商会在测试前先征求你的同意；同时提供识别和翻译的服务商（例如 Qwen Cloud、OpenAI）会先只测识别，点结果下方的 **Include translation in the test…** 可以把翻译也加进测试。
- Qwen Cloud 卡片上的 **Region** 可以选 **Singapore (international)** 或 **Beijing (mainland China)**，必须和你创建密钥的控制台一致。北京地域有免费额度。**Workspace ID** 是可选的，留空即可。
- 每个服务商的注册方法、免费额度和常见错误见 [云服务设置说明](docs/cloud-setup.md)（英文）。

![云服务设置](docs/screenshots/settings-cloud.png)

**AI assistant（AI 助手）**：AI 笔记和自动标题由你选择的聊天模型来写，费用计入你自己的 API 账号，识别和翻译不受影响。

- **Provider**：Qwen（阿里云百炼）、OpenAI、DeepSeek、Google Gemini、Groq、OpenRouter、硅基流动，或自定义的 OpenAI 兼容端点（例如本机运行的 Ollama）。先在 **Cloud providers** 里保存对应的密钥；自定义端点则要在那里填写 **Base URL**，密钥可以不填。
- **Model**：留空时使用默认模型（例如 `qwen-plus`、`gpt-4o-mini`）。长课程请选择上下文窗口更大的模型。
- **Test**：只发送一句固定的提示，不发送任何转录内容。
- **Consent**：打开 “Send transcripts and attached files to this model for notes and titles” 后，才允许把转录和附件文字发给这个模型。同意只针对当前选择的服务商，换一个 Provider 后需要重新同意。
- **Session titles**：打开 “Name sessions automatically when they end”（默认打开）并且已经同意发送后，至少 30 个词的课程会在下课时自动得到一个用目标语言写的标题。

![AI 助手设置](docs/screenshots/settings-assistant.png)

**同意对话框**：带小锁图标的按钮（例如选了云服务时的 **Start**，以及 **Create notes**、**Generate title**、翻译服务的 **Test**）表示这一步需要你的同意。点击后会弹出对话框，说明哪些内容会发给谁、哪些内容留在本机。对话框打开时焦点在不上传的按钮上（**Not now**，导入课件时是 **Extract on this computer**），只有你点了 **Allow…** 开头的按钮才会生效。同意会保存为默认设置，之后可以随时在 **Settings → Privacy** 或 **Settings → AI assistant** 里关闭。如果还缺少设置（比如没有保存密钥），对话框里会有按钮直接打开对应的设置页面。

![上传前的同意对话框](docs/screenshots/consent.png)

### 应用内更新

- EchoLingo 默认在每次启动时检查一次有没有新版本，可以在 **Settings → General → Updates** 关闭。
- **macOS 和 Windows** 可以直接在应用内更新：EchoLingo 下载新版本、校验它的签名，然后原地替换应用并重新启动。课程、设置、模型和密钥都会保留。
- **Linux**：有新版本时，EchoLingo 会提示并给出发布页面的链接。下载新的 .deb，像第一次安装时一样用 `sudo apt install` 安装即可，数据同样保留。
- **从 0.2.0 升级**：0.2.0 还没有应用内更新，需要手动安装一次：下载最新的 dmg，⌘Q 退出 EchoLingo，把新版本拖进“应用程序”文件夹替换旧版本。之后的版本都可以在应用内更新。
- 用了 GPU 加速包时，每次更新后都要重新下载加速包，见 [Windows / Linux](#windows--linux)。
- 更新后，macOS 可能会要求重新授予麦克风或“屏幕与系统录音”权限，见[没有字幕，或者听不到声音](#没有字幕或者听不到声音)。

## 隐私

**始终留在本机的内容**

- 音频的采集、降噪、增益、语音检测和重采样。
- 使用本机模型时的全部识别和翻译。
- 课程历史：转录、译文、AI 笔记和设置。
- 下课后用来校准时间戳的临时音频：校准成功、取消或失败后立即删除；应用异常退出时留下的文件，超过 24 小时后会在下次开始课程时清理。历史记录里不保存完整录音。

**会发送出去的内容（仅在你同意之后）**

| 功能 | 发送的内容 | 发送给 | 需要打开的开关 |
| --- | --- | --- | --- |
| 云端识别 | 本机处理后的课程音频，在课程进行期间持续上传；多数服务还会收到课程上下文或其中的术语，作为识别提示 | 你选择的识别服务 | Settings → Privacy → Audio upload |
| 云端翻译 | 识别出的原文句子；视服务而定，还可能包括少量前文（Qwen-MT 还包括之前的译文）、课程主题和术语对照 | 你选择的翻译服务 | Settings → Privacy → Transcript upload |
| AI 笔记和标题 | 这节课的转录、课程上下文、附件中提取的文字 | 你在 AI assistant 中选择的服务 | Settings → AI assistant → Consent |
| 从课件导入（AI 挑选术语） | 从课件中提取的文字 | 同上 | 同上 |
| 连接测试 | 识别：只做握手；翻译：一句固定英文；AI 助手：一句固定提示 | 被测试的服务 | 翻译测试需要 Transcript upload，其余无 |

AI 笔记和标题永远不会发送音频。

**检查更新**：EchoLingo 启动时从 GitHub 读取一个很小的版本清单（`latest.json`），更新包也从 GitHub 下载。检查更新只连接 GitHub，不会上传任何内容；可以在 **Settings → General → Updates** 关闭。

**如何撤销**：在 **Settings → Privacy** 关闭音频上传或文字上传；在 **Settings → AI assistant** 关闭 Consent，或把 Provider 设为 **Off**；在 **Settings → Cloud providers** 点 **Clear** 从系统安全存储里删除密钥。已经发送给第三方服务的数据，受该服务自己的隐私政策约束。

**数据存放位置**：
- macOS：`~/Library/Application Support/app.echolingo.desktop/`
- Windows：`%LOCALAPPDATA%\app.echolingo.desktop\`
- Linux：`~/.local/share/app.echolingo.desktop/`

| 路径 | 内容 |
| --- | --- |
| `history.sqlite` | 课程、转录、译文、AI 笔记 |
| `preferences.json` | 设置（不含密钥） |
| `models/` | 本地模型 |
| `runtimes/gpu-pack/` | GPU 加速包（Windows 和 Linux，安装后才有） |
| `alignment-spool/` | 时间戳校准用的临时音频 |
| `logs/` | 日志，用于排查问题 |

密钥保存在系统的安全存储中（macOS 钥匙串、Windows 凭据管理器或 Linux 的 Secret Service，服务名 `app.echolingo.desktop`），不会写进设置文件或日志。完全卸载时，先在 **Cloud providers** 中逐个点 **Clear** 删除密钥，再卸载应用（Windows 在“设置 → 应用 → 安装的应用”里卸载），并删除上面这个文件夹。

## 常见问题

### 模型下载很慢或失败

在中国大陆可以试试 Hugging Face 镜像。macOS 上运行 `launchctl setenv HF_ENDPOINT https://hf-mirror.com`，⌘Q 退出后重新打开 EchoLingo，再点 **Download**；Windows 和 Linux 的设置方法见 [Windows / Linux](#windows--linux)。如果下载没有走你的代理，也可以用同样的方式设置 `HTTPS_PROXY`，例如 macOS 上 `launchctl setenv HTTPS_PROXY http://127.0.0.1:7890`、Windows 上 `setx HTTPS_PROXY http://127.0.0.1:7890`（换成你自己的代理地址）。下载失败或中断后，再点一次 **Download** 即可。

### 点 Start 后提示模型未就绪（required local model is not ready）

说明本地模型还没有下载，请到 **Settings → Models** 下载。

### 第一次点 Start 要等很久

第一次加载模型需要 1 到 2 分钟。在 **Settings → Inference** 打开 “Preload local models when EchoLingo launches”（默认已打开）后，应用启动时会在后台加载好模型。

### 没有字幕，或者听不到声音

- macOS 上打开“系统设置 → 隐私与安全性 → 麦克风”，确认 EchoLingo 已开启。
- 采集电脑播放的声音时，还要在“系统设置 → 隐私与安全性 → 屏幕与系统录音”（macOS 14 上叫“屏幕录制”）中开启 EchoLingo，然后重新打开应用。
- 更新 Beta 版后，macOS 可能会要求重新授权。如果开关已经打开但仍然没有声音，可以先把 EchoLingo 从列表中移除，再重新授权。
- Windows 上打开“设置 → 隐私和安全性 → 麦克风”，确认“麦克风访问权限”和“允许桌面应用访问你的麦克风”都已打开。
- Linux 上在系统的声音设置里检查输入设备和音量。
- 看一下 **Live** 右侧 **Audio / VAD** 面板的音量条有没有跳动。

### 字幕越来越慢（Windows / Linux）

没有安装 GPU 加速包时，识别在 CPU 上运行，较慢的处理器可能跟不上实时速度。如果电脑有支持的 NVIDIA 显卡，可以安装 [GPU 加速包](#windows--linux)；也可以在 **Settings → Cloud providers** 设置一个云端识别服务：推理模式为 Auto 并允许音频上传后，本机模型达不到延迟目标时会改用它。

### 悬浮字幕窗没有保持在最上层（Linux）

在 Wayland 桌面上，应用不一定能让自己的窗口始终置顶。可以用 X11 后端启动 EchoLingo，例如 `GDK_BACKEND=x11 echolingo-desktop`，或者登录时选择 X11 会话。

### 无法保存密钥（Linux）

EchoLingo 通过 Secret Service 保存密钥。请安装或解锁 GNOME 钥匙环或 KWallet，然后重新打开 EchoLingo。在此之前，可以在启动应用前设置服务商对应的环境变量（例如 `DEEPL_API_KEY`），**Settings → Cloud providers** 的每张卡片上都写明了变量名。

### Qwen Cloud 测试失败

- HTTP 401：通常是 **Region** 和密钥不匹配。在[百炼控制台](https://bailian.console.aliyun.com)（北京）创建的密钥要选 **Beijing**，在[国际站 Model Studio 控制台](https://modelstudio.console.alibabacloud.com)创建的要选 **Singapore**。
- 与工作空间相关的 HTTP 403：清空 **Workspace ID**，使用密钥的默认工作空间。
- HTTP 403 `AllocationQuota.FreeTierOnly`：免费额度已经用完。
- 在中国大陆，百炼北京地域、DeepSeek 和硅基流动可以直接访问；OpenAI、Deepgram、AssemblyAI、Gladia、DeepL、Google 和 Azure 通常需要代理。

### AI 笔记报错

| 提示（英文界面） | 含义 | 解决方法 |
| --- | --- | --- |
| The AI assistant is not set up yet. | 还没有选择 AI 助手或保存密钥 | 在 **Settings → AI assistant** 选择 Provider，并在 **Cloud providers** 保存密钥 |
| Sending transcripts to the AI assistant is turned off. | 没有允许发送转录 | 在 **Settings → AI assistant** 打开 Consent |
| The provider rejected the API key. | 密钥被拒绝 | 在 **Cloud providers** 检查密钥（Qwen 还要检查 Region） |
| The provider’s rate limit or quota was reached. | 触发了服务商的速率限制或额度上限 | 等一会儿再试，或者换一个服务商 |
| This session is too long for the selected model. | 课程太长，超出了模型的上下文窗口 | 换一个上下文更大的模型，或者少附几个文件 |
| EchoLingo could not reach the provider. | 连不上服务商 | 检查网络和代理 |

出错时旧的笔记不会丢失。错误提示里通常有按钮，可以直接打开对应的设置页面，或者点 **Try again** 重试。

### 识别不准怎么办

- 上课前填写**课程上下文**：主题、人名、专业术语，写得越具体越好。也可以用 **Import from slides…** 从课件中导入。
- 把长期用到的术语和固定译法写进 **Settings → Translation → Glossary**。
- 麦克风尽量朝向讲台或教室扬声器，远离空调出风口和电脑风扇；条件允许的话用外接麦克风。音频配置保持 **Lecture / far-field**。
- 确认源语言选对了。网课和视频请用 **System audio**，不要让麦克风去收扬声器外放的声音。

### 能完全离线使用吗？

可以。模型下载完成后，本机识别和翻译都不需要网络。AI 笔记需要联网，除非你把 AI 助手设成本机运行的自定义端点（例如 Ollama）。

## 路线图

- 支持 AMD 和 Intel 显卡的 GPU 加速。
- Linux 上的系统声音采集，以及适用于更多发行版的 AppImage 包。
- Windows 代码签名和 Apple 公证，免去首次打开时的手动步骤。
- Windows 和 Linux 的 ARM64 版本。
- 支持更多语言。

## 从源码构建

需要 Node.js 22 LTS、Rust 1.88 及以上（stable）和 Conda（Miniconda 或 Miniforge），以及：

- **macOS**：macOS 14 及以上的 Apple Silicon Mac，以及 Xcode Command Line Tools。
- **Windows**：Windows 10 或 11（x64），Visual Studio 2022 生成工具（选“使用 C++ 的桌面开发”）和 WebView2 运行时。
- **Linux**：Ubuntu 22.04 及以上（x64），并运行 `sudo apt install build-essential curl file libwebkit2gtk-4.1-dev libayatana-appindicator3-dev librsvg2-dev libxdo-dev libssl-dev libasound2-dev libdbus-1-dev pkg-config`。

```bash
git clone https://github.com/Asphr726/EchoLingo.git
cd EchoLingo

# Python 环境和语音识别运行时
conda env create -f environment.yml
conda activate echolingo-spike1
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu   # 仅 Windows 和 Linux：安装 CPU 版 PyTorch
python -m pip install -e ".[qwen]"

# 翻译运行时（当前系统对应的 llama.cpp）
python scripts/fetch_llama_runtime.py

# 桌面应用
npm install
python scripts/build_sidecar.py --clean          # 打包内置推理服务（同时下载固定版本的 Silero VAD 模型）
npm run desktop:dev                              # 开发模式运行
npm run desktop:build                            # Windows 和 Linux：生成安装包
APPLE_SIGNING_IDENTITY=- npm run desktop:build   # macOS：本机自签名构建
```

构建产物位于 `target/release/bundle/`：macOS 是 `dmg/EchoLingo_0.3.1_aarch64.dmg`，Windows 是 `nsis/EchoLingo_0.3.1_x64-setup.exe`，Linux 是 `deb/EchoLingo_0.3.1_amd64.deb`。`APPLE_SIGNING_IDENTITY=-` 生成的是只适合自己使用的临时签名；公开分发需要 Developer ID 证书和 Apple 公证。

运行测试：

```bash
pytest                                      # Python（需要先激活 conda 环境）
cargo test --workspace                      # Rust
npm run typecheck && npm run test:desktop   # 界面
```

### 发布新版本（维护者）

应用内更新只接受用 EchoLingo 更新签名密钥签过名的更新包。私钥约定保存在 `~/.tauri/echolingo-updater.key`（用 `npx tauri signer generate -w ~/.tauri/echolingo-updater.key` 生成，公钥是 `apps/desktop/src-tauri/tauri.conf.json` 里的 `plugins.updater.pubkey`）。请务必离线备份这个文件：丢失后，已安装的应用再也无法在应用内更新，只能手动重装。本地构建不需要私钥；CI 从仓库 secret `TAURI_SIGNING_PRIVATE_KEY` 读取私钥内容（私钥有密码时另设 `TAURI_SIGNING_PRIVATE_KEY_PASSWORD`），例如 `gh secret set TAURI_SIGNING_PRIVATE_KEY < ~/.tauri/echolingo-updater.key`。

1. 更新版本号，写好 `.github/release-notes/vX.Y.Z.md`，推送标签 `vX.Y.Z`。
2. CI 构建全部平台，并创建一个草稿 `vX.Y.Z`：Mac 版 dmg 和更新包 `.app.tar.gz`、Windows 安装程序、Linux .deb（后三者各带 `.sig` 签名）、GPU 加速包、`SHA256SUMS.txt`（校验值也附在发布说明末尾），以及供检查的 `latest.json`。
3. 检查无误后，把 `vX.Y.Z` 发布为 pre-release。
4. 运行 `python scripts/update_manifest.py --tag vX.Y.Z --publish`（需要已登录的 GitHub CLI `gh`）。它会列出这个版本里带签名的 macOS、Windows 和 Linux 更新文件，校验签名密钥和下载链接，然后替换 `updater` 这个 pre-release 上的 `latest.json`，已安装的应用下次检查时就会看到新版本。

`python scripts/smoke_update.py` 可以用临时密钥在本机 Mac 上完整演练一次应用内更新。

## 许可与致谢

EchoLingo 以 [Apache License 2.0](LICENSE) 开源。本地模型不随应用分发，由你在应用内从 Hugging Face 下载，遵循各自的许可证。完整的第三方组件和许可证列表见 [docs/third-party-components.md](docs/third-party-components.md)。

感谢这些项目：

- [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)（通义千问团队），语音识别和强制对齐模型
- [Hy-MT2](https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF)（腾讯混元），本地翻译模型
- [WhisperLiveKit](https://github.com/QuentinFuxa/WhisperLiveKit)，流式识别框架
- [llama.cpp](https://github.com/ggml-org/llama.cpp)，本地翻译模型的推理运行时
- [Silero VAD](https://github.com/snakers4/silero-vad)，语音检测
- [Tauri](https://tauri.app)，桌面应用框架
