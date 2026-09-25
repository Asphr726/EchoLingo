/** In-app deep links between main-window views ("Open AI assistant
 *  settings" from History, "Add a key in Cloud providers" from Settings).
 *
 *  A request is a cancelable `echolingo:navigate` window event whose detail
 *  is a `NavigationTarget`. The main window switches view on it, and an open
 *  Settings view switches section; a handler calls `preventDefault()` so the
 *  caller can tell the request was taken. The settings section (and the
 *  element to focus in it) is also remembered here so a freshly mounted
 *  Settings view opens on it. */

export type AppView = "live" | "history" | "settings";

export interface NavigationTarget {
  view: AppView;
  /** Settings section slug, e.g. `ai-assistant` or `cloud-providers`. */
  section?: string;
  /** Id of the element in the section to focus, e.g. `settings-updates`;
   *  without one the section heading takes focus. */
  anchor?: string;
}

export const NAVIGATE_EVENT = "echolingo:navigate";

const views: readonly AppView[] = ["live", "history", "settings"];

let pendingSection: string | null = null;
let pendingAnchor: string | null = null;

export function isAppView(value: unknown): value is AppView {
  return typeof value === "string" && (views as readonly string[]).includes(value);
}

/** Asks the main window to show `target`. Returns true when a view took it. */
export function requestNavigation(target: NavigationTarget): boolean {
  if (typeof window === "undefined") return false;
  pendingSection = target.view === "settings" ? target.section ?? null : null;
  pendingAnchor = target.view === "settings" ? target.anchor ?? null : null;
  const event = new CustomEvent<NavigationTarget>(NAVIGATE_EVENT, { detail: target, cancelable: true });
  return !window.dispatchEvent(event);
}

/** Calls `handler` for every valid navigation request; returns the
 *  unsubscribe function (an effect cleanup). */
export function onNavigate(handler: (target: NavigationTarget) => void): () => void {
  if (typeof window === "undefined") return () => undefined;
  const listener = (event: Event) => {
    const detail = (event as CustomEvent<NavigationTarget | undefined>).detail;
    if (!detail || !isAppView(detail.view)) return;
    event.preventDefault();
    handler(detail);
  };
  window.addEventListener(NAVIGATE_EVENT, listener);
  return () => window.removeEventListener(NAVIGATE_EVENT, listener);
}

/** The section requested by the last navigation. Reading has no side
 *  effect (React may run state initializers twice); the view that opened
 *  it calls `clearPendingSection` from an effect. */
export function peekPendingSection(): string | null {
  return pendingSection;
}

/** The element the last navigation asked Settings to focus. */
export function peekPendingAnchor(): string | null {
  return pendingAnchor;
}

/** Forgets the pending section and its anchor. */
export function clearPendingSection(): void {
  pendingSection = null;
  pendingAnchor = null;
}
