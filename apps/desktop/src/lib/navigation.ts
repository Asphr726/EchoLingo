/** In-app deep links between main-window views ("Open AI assistant
 *  settings" from History, "Add a key in Cloud providers" from Settings).
 *
 *  A request is a cancelable `echolingo:navigate` window event whose detail
 *  is a `NavigationTarget`. The main window handles it (and calls
 *  `preventDefault()`); until it does, the request falls back to pressing the
 *  matching primary-navigation button. The settings section is remembered
 *  here so a freshly mounted Settings view opens on it. */

export type AppView = "live" | "history" | "settings";

export interface NavigationTarget {
  view: AppView;
  /** Settings section slug, e.g. `ai-assistant` or `cloud-providers`. */
  section?: string;
}

export const NAVIGATE_EVENT = "echolingo:navigate";

let pendingSection: string | null = null;

export function requestNavigation(target: NavigationTarget): void {
  if (typeof window === "undefined") return;
  pendingSection = target.section ?? null;
  const event = new CustomEvent<NavigationTarget>(NAVIGATE_EVENT, { detail: target, cancelable: true });
  const handled = !window.dispatchEvent(event);
  if (handled) return;
  const label = target.view.toLowerCase();
  const button = Array.from(document.querySelectorAll<HTMLButtonElement>(".rail-link")).find(
    (candidate) => candidate.textContent?.trim().toLowerCase() === label,
  );
  button?.click();
}

/** The section requested by the last navigation. Reading has no side
 *  effect (React may run state initializers twice); the view that opened
 *  it calls `clearPendingSection` from an effect. */
export function peekPendingSection(): string | null {
  return pendingSection;
}

export function clearPendingSection(): void {
  pendingSection = null;
}
