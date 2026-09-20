import { DotsSixVertical, X } from "@phosphor-icons/react";
import { type CSSProperties, useLayoutEffect, useRef } from "react";
import { api } from "../lib/bridge";
import { useApp } from "../state/AppContext";

export function CaptionWindow() {
  const { caption, loading, snapshot } = useApp();
  const recent = snapshot.previous_segments.slice(-caption.recent_segments);
  const showOriginal = caption.display !== "translation";
  const showTranslation = caption.display !== "original";
  const openText = snapshot.live.open_text.trim();
  const unstableText = snapshot.live.original_unstable.trim();
  const provisional = openText || unstableText ? snapshot.live.translation_editable.trim() : "";
  const hasLive = Boolean(openText || unstableText);
  const sourceLanguage = snapshot.config?.source_language;
  const targetLanguage = snapshot.config?.target_language;
  const joiner = sourceLanguage === "zh" || sourceLanguage === "ja" ? "" : " ";
  const copyRef = useRef<HTMLElement>(null);
  const contentToken = `${recent.map((segment) => `${segment.id}:${segment.translation}`).join("|")}:${openText}:${unstableText}:${provisional}`;

  // The caption always shows the newest text: keep the tail anchored.
  useLayoutEffect(() => {
    const copy = copyRef.current;
    if (copy) copy.scrollTop = copy.scrollHeight;
  }, [contentToken]);

  return (
    <main
      className="caption-window"
      style={
        {
          "--caption-opacity": caption.opacity,
          "--caption-font-size": `${caption.font_size_px}px`,
        } as CSSProperties
      }
    >
      <header className="caption-drag" data-tauri-drag-region>
        <div className="caption-handle" data-tauri-drag-region>
          <DotsSixVertical size={18} weight="bold" aria-hidden="true" />
          <span data-tauri-drag-region>EchoLingo</span>
        </div>
        <span className={`caption-state caption-state--${snapshot.phase.toLowerCase()}`}>
          {snapshot.phase === "LISTENING" ? "Live" : snapshot.phase.toLowerCase()}
        </span>
        <button
          className="icon-button caption-close"
          type="button"
          aria-label="Hide floating caption"
          onClick={() => void api.hideCaption()}
        >
          <X size={17} weight="bold" aria-hidden="true" />
        </button>
      </header>
      <section className="caption-copy" ref={copyRef}>
        {loading ? (
          <div className="caption-skeleton" aria-label="Loading caption" />
        ) : !hasLive && recent.length === 0 ? (
          <p className="caption-empty">
            Captions will appear here when a session starts.
          </p>
        ) : (
          <>
            {recent.map((segment) => (
              <div className="caption-segment caption-segment--previous" key={segment.id}>
                {showOriginal && <p lang={sourceLanguage}>{segment.original}</p>}
                {showTranslation && segment.translation && (
                  <p className="caption-translation" lang={targetLanguage}>
                    {segment.translation}
                  </p>
                )}
              </div>
            ))}
            {hasLive && (
              <div className="caption-segment caption-segment--current">
                {showOriginal && (
                  <p lang={sourceLanguage}>
                    {openText && <span className="text-open">{openText}</span>}
                    {openText && unstableText ? joiner : ""}
                    {unstableText && <span className="text-unstable">{unstableText}</span>}
                  </p>
                )}
                {showTranslation && provisional && (
                  <p className="caption-translation caption-translation--provisional" lang={targetLanguage}>
                    {provisional}
                  </p>
                )}
              </div>
            )}
          </>
        )}
      </section>
    </main>
  );
}
