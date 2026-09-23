/** Browser-preview fixtures for History, AI notes and the lecture-context
 *  import (`?preview=live&view=history`). Nothing here runs in the desktop
 *  runtime and no network is touched.
 *
 *  `?notes=` selects a state of the AI notes panel for review:
 *  `saved` (default) · `empty` · `streaming` · `sheet` · `error` ·
 *  `unconfigured`. */

import type {
  AssistantJob,
  AssistantProbeResult,
  AssistantStatus,
  ContextImportResult,
  NoteAttachment,
  SegmentRecord,
  SessionDetail,
  SessionNotes,
  SessionNotesState,
  SessionRecord,
} from "../types";

export type PreviewNotesMode = "saved" | "empty" | "streaming" | "sheet" | "error" | "unconfigured";

export function previewNotesMode(): PreviewNotesMode {
  const requested = new URLSearchParams(window.location.search).get("notes");
  const modes: PreviewNotesMode[] = ["saved", "empty", "streaming", "sheet", "error", "unconfigured"];
  return modes.find((mode) => mode === requested) ?? "saved";
}

export const PREVIEW_NOTES_SESSION = "7c1f4e0a-preview-notes";

// Notes as the assistant writes them for an en → zh lecture: in the target
// language, standard terms kept in English on first use.
export const previewNotesMarkdown = String.raw`# 颜色空间与前注意纹理感知

本讲从 RGB 立方体出发，解释 HSV 为什么把色相排成一个六边形，再转向 Béla Julesz 的纹理分割实验：哪些差异能在一瞥之间“跳出来”，哪些必须逐个扫视才能找到。最后讨论 texton 理论与 Treisman 的特征整合理论，并布置了作业 2。

## RGB 立方体与色相六边形 (00:00–08:40)

- 沿白—黑对角线（灰轴）俯视 **RGB 立方体**，其余六个角投影成一个六边形：红、黄、绿、青、蓝、品红。
- 从左下角的红色出发，经橙色、黄色到绿色，色相沿六边形的边连续变化。
- 与灰轴**正交**的方向对应饱和度：离灰轴越远，颜色越“纯”。

## HSV 的计算 (08:40–21:15)

HSV 用三个量描述颜色：色相 $H$、饱和度 $S$ 和明度 $V$。设 $M = \max(R,G,B)$、$m = \min(R,G,B)$，色度 $C = M - m$：

\[
V = M, \qquad S = \begin{cases} C / M & M > 0 \\ 0 & M = 0 \end{cases}
\]

| 量 | 取值范围 | 直观含义 |
| --- | --- | --- |
| $H$ | $[0^\circ, 360^\circ)$ | 在六边形上的角度 |
| $S$ | $[0, 1]$ | 离灰轴的距离 |
| $V$ | $[0, 1]$ | 最亮通道的强度 |

讲者提醒：当 $C = 0$ 时色相无定义，实现时通常记为 0。

## 前注意视觉与纹理分割 (21:15–34:50)

**前注意视觉**（pre-attentive vision）指无需集中注意、约 200 ms 内即可完成的加工。

1. Julesz 的实验：两种纹理的一阶统计量（平均亮度）相同时，二阶统计量不同的纹理仍能立即分开。
2. 反例：某些二阶统计量相同的纹理也能被区分，Julesz 因此转向 **texton**（纹理基元）：线段末端、交叉点和朝向。
3. 讲者用 “L” 与 “T” 的阵列演示：两者由相同的线段组成，却很难一眼分开。

## 视觉搜索与特征整合 (34:50–51:30)

- 目标与干扰项只差一个特征（颜色或朝向）时，反应时与干扰项数量无关，目标会“跳出来”。
- 需要组合两个特征时，反应时随数量线性增长：\( RT = a + b\,n \)，其中 $b$ 约为每项 25–50 ms (?)。
- Treisman 的**特征整合理论**：注意把分散在各特征图中的信息“粘合”起来。
- 眼跳（saccade）约每秒 3–4 次，每次注视约 250 ms。

## 作业与下周安排 (51:30–64:05)

- 作业 2：实现 RGB → HSV 转换，并用色相直方图做简单的图像检索，下周五截止。
- 下周阅读 Malik 与 Perona 关于纹理感知的论文，讲者口述的年份是 1990 年 (?)。
- 课程主页：https://inst.eecs.berkeley.edu/~cs180/

## 关键术语

- **前注意视觉** pre-attentive vision
- **眼跳** saccade
- **纹理基元** texton
- **特征整合理论** feature integration theory
`;

const lines: Array<[string, string]> = [
  ["Look at that line and think about what's happening in the orthogonal direction to it.", "看看那条线，思考一下在垂直方向上会发生什么。"],
  ["It's gonna wrap around the six corners of the cube, the hexagon of the cube away from white.", "它会围绕立方体的六个角，也就是从白色区域向外延伸的六边形区域。"],
  ["Okay, so starting at the lower left corner here, it's red.", "好的，那么从这里的左下角开始，是红色的。"],
  ["So value is just the max of the three channels, and saturation divides the chroma by it.", "所以明度就是三个通道中的最大值，饱和度是色度除以它。"],
  ["If the chroma is zero, the hue is undefined, so people just call it zero.", "如果色度为零，色相就没有定义，所以大家通常记为零。"],
  ["Julesz asked which textures you can segment without scrutiny, in a glance.", "Julesz 提出的问题是：哪些纹理你不需要仔细看，一眼就能分开。"],
  ["Same first-order statistics, same mean brightness, and yet they pop out.", "一阶统计量相同，平均亮度相同，但它们还是会跳出来。"],
  ["These L's and T's are made of exactly the same line segments.", "这些 L 和 T 由完全相同的线段组成。"],
  ["When the target differs in one feature, the reaction time is flat.", "当目标只在一个特征上不同时，反应时是平的。"],
  ["Conjunction search goes up linearly with the number of items.", "联合搜索的反应时随项目数量线性上升。"],
  ["You make about three or four saccades every second.", "你每秒大约会做三到四次眼跳。"],
  ["Homework two is due next Friday; it's the RGB to HSV conversion.", "作业二下周五截止，内容是 RGB 到 HSV 的转换。"],
];

function previewSegments(): SegmentRecord[] {
  // Spread over a 64-minute lecture so section time chips land on rows.
  const starts = [12, 250, 505, 640, 1_190, 1_300, 1_720, 2_050, 2_110, 2_600, 3_020, 3_110];
  return lines.map(([source, target], index) => ({
    id: `preview-row-${index}`,
    start_ms: starts[index] * 1000,
    end_ms: starts[index] * 1000 + 6_400,
    source_text: source,
    translated_text: target,
    timestamp_quality: "forced",
  }));
}

const hoursAgo = (hours: number) => new Date(Date.now() - hours * 3_600_000).toISOString();

const records: SessionRecord[] = [
  {
    id: PREVIEW_NOTES_SESSION,
    title: "颜色空间与前注意纹理感知",
    title_source: "ai",
    context: "CS180 lecture 12: colour spaces and texture perception\nBéla Julesz\nsaccade = 眼跳",
    status: "completed",
    started_at: hoursAgo(3),
    ended_at: hoursAgo(1.93),
    source_language: "en",
    target_language: "zh",
    audio_source: "microphone",
    audio_profile: "lecture",
    inference_mode: "auto",
    asr_backend: "qwen_local",
    translation_backend: "hymt_local",
    route_reason: "preview",
  },
  {
    id: "3a9d2b61-preview-default",
    title: "en → zh lecture",
    title_source: "default",
    context: "",
    status: "completed",
    started_at: hoursAgo(27),
    ended_at: hoursAgo(26),
    source_language: "en",
    target_language: "zh",
    audio_source: "microphone",
    audio_profile: "lecture",
    inference_mode: "auto",
    asr_backend: "qwen_local",
    translation_backend: "hymt_local",
    route_reason: "preview",
  },
  {
    id: "b04e77c8-preview-user",
    title: "CS180 week 5: filtering",
    title_source: "user",
    context: "",
    status: "completed",
    started_at: hoursAgo(75),
    ended_at: hoursAgo(74),
    source_language: "en",
    target_language: "zh",
    audio_source: "system_audio",
    audio_profile: "lecture",
    inference_mode: "cloud",
    asr_backend: "qwen_cloud",
    translation_backend: "qwen_cloud",
    route_reason: "preview",
  },
];

export const previewAttachments: NoteAttachment[] = [
  { id: "att-slides", name: "CS180 Lecture 12 – Colour and texture.pdf", size_bytes: 4_812_300, extension: "pdf" },
  { id: "att-handout", name: "texture-perception-handout.pptx", size_bytes: 1_204_000, extension: "pptx" },
  { id: "att-reading", name: "julesz-1981-textons.md", size_bytes: 38_400, extension: "md" },
];

export const previewContextImport: ContextImportResult = {
  context: "CS180 Lecture 12: colour and texture\nBéla Julesz",
  terms: ["texton", "Anne Treisman", "conjunction search"],
  glossary: [
    { source: "pre-attentive vision", target: "前注意视觉" },
    { source: "feature integration theory", target: "特征整合理论" },
  ],
  warnings: ["Slides 14–16 are images without text; add their terms by hand."],
};

type Emit = (kind: "assistant_update" | "history_changed", payload: unknown, sessionId: string | null) => void;

/** In-memory History and assistant for the preview. */
export class PreviewAssistant {
  private readonly mode = previewNotesMode();
  private notes = new Map<string, SessionNotes>();
  private jobs = new Map<string, AssistantJob>();
  private timers = new Map<string, number>();
  private picks = 0;
  private sessions = records.map((record) => ({ ...record }));
  private streamingStarted = false;

  constructor(private readonly emit: Emit) {
    if (this.mode === "saved" || this.mode === "error") {
      this.notes.set(PREVIEW_NOTES_SESSION, savedNotes(PREVIEW_NOTES_SESSION));
    }
    if (this.mode === "error") {
      this.jobs.set(PREVIEW_NOTES_SESSION, {
        job_id: "preview-failed",
        session_id: PREVIEW_NOTES_SESSION,
        task: "notes",
        state: "failed",
        error: { code: "authentication_failed", message: "Qwen (Alibaba Model Studio) rejected the API key (HTTP 401)." },
      });
    }
  }

  status(): AssistantStatus {
    const configured = this.mode !== "unconfigured";
    return {
      configured,
      consent: configured,
      provider_group: configured ? "dashscope" : "",
      model: configured ? "qwen-plus" : "",
      display_name: configured ? "Qwen (Alibaba Model Studio)" : "",
      key_available: configured,
      auto_title: true,
    };
  }

  probe(): AssistantProbeResult {
    return { provider: "dashscope", model: "qwen-plus", latency_ms: 842 };
  }

  search(query: string): SessionRecord[] {
    const needle = query.trim().toLowerCase();
    return this.sessions.filter((session) => !needle || session.title.toLowerCase().includes(needle));
  }

  open(sessionId: string): SessionDetail {
    const session = this.sessions.find((record) => record.id === sessionId) ?? this.sessions[0];
    return { session: { ...session }, segments: previewSegments() };
  }

  sessionNotes(sessionId: string): SessionNotesState {
    if (this.mode === "streaming" && sessionId === PREVIEW_NOTES_SESSION && !this.streamingStarted) {
      this.streamingStarted = true;
      this.startJob(sessionId, Math.round(previewNotesMarkdown.length * 0.46));
    }
    return { notes: this.notes.get(sessionId) ?? null, job: this.jobs.get(sessionId) ?? null };
  }

  pick(): NoteAttachment[] {
    const picked = previewAttachments[this.picks % previewAttachments.length];
    this.picks += 1;
    return [picked];
  }

  create(sessionId: string): { job_id: string } {
    return { job_id: this.startJob(sessionId, 0) };
  }

  cancel(jobId: string): void {
    for (const [sessionId, job] of this.jobs) {
      if (job.job_id !== jobId || job.state !== "running") continue;
      window.clearInterval(this.timers.get(jobId));
      const cancelled: AssistantJob = { ...job, state: "cancelled" };
      this.jobs.set(sessionId, cancelled);
      this.emit("assistant_update", cancelled, sessionId);
    }
  }

  title(sessionId: string): { title: string } {
    const session = this.sessions.find((record) => record.id === sessionId);
    const title = "卷积、滤波与图像金字塔";
    if (session && session.title_source !== "user") {
      session.title = title;
      session.title_source = "ai";
      window.setTimeout(() => this.emit("history_changed", { session_id: sessionId, reason: "title" }, sessionId), 0);
    }
    return { title: session?.title ?? title };
  }

  private startJob(sessionId: string, fromChars: number): string {
    const jobId = `preview-job-${Date.now().toString(36)}`;
    const total = 5;
    const ranges: Array<[number, number]> = [
      [0, 520_000],
      [520_000, 1_275_000],
      [1_275_000, 2_090_000],
      [2_090_000, 3_090_000],
      [3_090_000, 3_845_000],
    ];
    let cursor = fromChars;
    const update = (state: AssistantJob["state"]) => {
      const done = cursor / previewNotesMarkdown.length;
      const index = Math.min(total, Math.floor(done * total) + 1);
      const [start_ms, end_ms] = ranges[index - 1];
      const job: AssistantJob = {
        job_id: jobId,
        session_id: sessionId,
        task: "notes",
        state,
        stage: state === "running" ? (cursor === 0 ? "reading" : done > 0.94 ? "finishing" : "section") : null,
        progress: { index, total, start_ms, end_ms },
        text: previewNotesMarkdown.slice(0, cursor),
        error: null,
      };
      this.jobs.set(sessionId, job);
      return job;
    };
    update("running");
    const timer = window.setInterval(() => {
      cursor = Math.min(previewNotesMarkdown.length, cursor + 14);
      if (cursor >= previewNotesMarkdown.length) {
        window.clearInterval(timer);
        this.notes.set(sessionId, savedNotes(sessionId));
        const finished = update("completed");
        this.jobs.delete(sessionId);
        this.emit("assistant_update", finished, sessionId);
        this.emit("history_changed", { session_id: sessionId, reason: "notes" }, sessionId);
        return;
      }
      this.emit("assistant_update", update("running"), sessionId);
    }, 110);
    this.timers.set(jobId, timer);
    return jobId;
  }
}

function savedNotes(sessionId: string): SessionNotes {
  const updated = new Date(Date.now() - 95 * 60_000).toISOString();
  return {
    session_id: sessionId,
    markdown: previewNotesMarkdown,
    language: "zh",
    provider: "dashscope",
    model: "qwen-plus",
    attachments: [
      { id: "att-slides", name: "CS180 Lecture 12 – Colour and texture.pdf", pages: 42, chars: 18_240, truncated: false, warning: null },
      { id: "att-handout", name: "texture-perception-handout.pptx", pages: 9, chars: 3_120, truncated: false, warning: "Slides 4–6 have no extractable text." },
    ],
    usage: { prompt_tokens: 21_406, completion_tokens: 2_874 },
    prompt_version: "notes.v1",
    source_chars: 58_310,
    created_at: updated,
    updated_at: updated,
  };
}
