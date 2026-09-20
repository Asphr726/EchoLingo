import {
  ClosedCaptioning,
  ClockCounterClockwise,
  GearSix,
  Waveform,
  X,
} from "@phosphor-icons/react";
import { useState } from "react";
import { useApp } from "../state/AppContext";
import { HistoryView } from "./HistoryView";
import { LiveView } from "./LiveView";
import { Onboarding } from "./Onboarding";
import { SettingsView } from "./SettingsView";

type View = "live" | "history" | "settings";

const views = [
  { id: "live" as const, label: "Live", icon: Waveform },
  { id: "history" as const, label: "History", icon: ClockCounterClockwise },
  { id: "settings" as const, label: "Settings", icon: GearSix },
];

export function MainWindow() {
  const [view, setView] = useState<View>(() => {
    const requested = new URLSearchParams(window.location.search).get("view");
    return requested === "history" || requested === "settings" ? requested : "live";
  });
  const { clearError, error, loading, onboardingComplete, snapshot } = useApp();
  const activeRoute = snapshot.route?.deployment ?? "not routed";

  if (!loading && !onboardingComplete) {
    return <Onboarding />;
  }

  return (
    <main className="desktop-shell">
      <aside className="rail" aria-label="Primary navigation">
        <div className="brand-mark" aria-label="EchoLingo">
          <span>EL</span>
        </div>
        <nav className="rail-nav">
          {views.map(({ id, icon: Icon, label }) => (
            <button
              className={`rail-link ${view === id ? "rail-link--active" : ""}`}
              key={id}
              type="button"
              aria-current={view === id ? "page" : undefined}
              onClick={() => setView(id)}
            >
              <Icon size={20} weight={view === id ? "fill" : "regular"} aria-hidden="true" />
              <span>{label}</span>
            </button>
          ))}
        </nav>
        <div className="rail-caption">
          <ClosedCaptioning size={17} weight="regular" aria-hidden="true" />
          <span>Lecture mode</span>
        </div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">EchoLingo / {view}</p>
            <h1>{view === "live" ? "Lecture interpreter" : capitalize(view)}</h1>
          </div>
          <div className="topbar-status" aria-label="Application status">
            <span className={`status-pulse status-pulse--${snapshot.phase.toLowerCase()}`} />
            <div>
              <strong>{loading ? "Connecting" : phaseLabel(snapshot.phase)}</strong>
              <span>{activeRoute.replaceAll("_", " ")}</span>
            </div>
          </div>
        </header>

        {error && (
          <div className="error-banner" role="alert">
            <div>
              <strong>Action needs attention</strong>
              <span>{error}</span>
            </div>
            <button className="icon-button" type="button" aria-label="Dismiss error" onClick={clearError}>
              <X size={17} weight="bold" aria-hidden="true" />
            </button>
          </div>
        )}

        <div className="view-stage">
          {view === "live" && <LiveView />}
          {view === "history" && <HistoryView />}
          {view === "settings" && <SettingsView />}
        </div>
      </section>
    </main>
  );
}

function capitalize(value: string) {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function phaseLabel(phase: string) {
  const labels: Record<string, string> = {
    IDLE: "Ready",
    STARTING: "Starting",
    LISTENING: "Listening",
    PAUSED: "Paused",
    STOPPING: "Stopping",
    COMPLETED: "Completed",
  };
  return labels[phase] ?? phase;
}
