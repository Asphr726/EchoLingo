import { DotsSixVertical, X } from "@phosphor-icons/react";
import type { CSSProperties } from "react";
import { api } from "../lib/bridge";
import { useApp } from "../state/AppContext";

export function CaptionWindow() {
  const { caption, loading, snapshot } = useApp();
  const recent = snapshot.previous_segments.slice(-caption.recent_segments);
  const showOriginal = caption.display !== "translation";
  const showTranslation = caption.display !== "original";
  const original = [snapshot.live.original_committed, snapshot.live.original_unstable]
    .filter(Boolean)
    .join(" ");
  const translation = [
    snapshot.live.translation_committed,
    snapshot.live.translation_editable,
  ]
    .filter(Boolean)
    .join(" ");

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
      <section className="caption-copy" aria-live="polite">
        {loading ? (
          <div className="caption-skeleton" aria-label="Loading caption" />
        ) : !original && !translation && recent.length === 0 ? (
          <p className="caption-empty">
            Captions will appear here when a session starts.
          </p>
        ) : (
          <>
            {recent.map((segment) => (
              <div className="caption-segment caption-segment--previous" key={segment.id}>
                {showOriginal && <p lang={snapshot.config?.source_language}>{segment.original}</p>}
                {showTranslation && segment.translation && (
                  <p className="caption-translation" lang={snapshot.config?.target_language}>
                    {segment.translation}
                  </p>
                )}
              </div>
            ))}
            <div className="caption-segment caption-segment--current">
              {showOriginal && original && (
                <p lang={snapshot.config?.source_language}>{original}</p>
              )}
              {showTranslation && translation && (
                <p className="caption-translation" lang={snapshot.config?.target_language}>
                  {translation}
                </p>
              )}
            </div>
          </>
        )}
      </section>
    </main>
  );
}
