/**
 * Scroll bookkeeping for the live transcript. While the reader follows the
 * feed it stays pinned to the newest line. Once they scroll back to read, the
 * line they are on keeps its place while rows arrive below, translations fill
 * in above, or the oldest rows drop off the 200-row window. WebKit has no
 * scroll anchoring, so the feed does this itself on every engine.
 */

/** Scrolling down to within this distance of the bottom resumes following. */
export const RESUME_FOLLOW_PX = 48;

export interface FeedMetrics {
  scrollTop: number;
  scrollHeight: number;
  clientHeight: number;
}

export function distanceFromBottom({ scrollTop, scrollHeight, clientHeight }: FeedMetrics): number {
  return Math.max(0, scrollHeight - scrollTop - clientHeight);
}

export type FollowChange = "follow" | "unfollow" | null;

/** How a scroll made by the reader (not by the feed itself) changes following.
 *  An upward move that still ends at the bottom is the browser clamping the
 *  position after the content shrank (a font swap, a shorter live line), not
 *  the reader leaving. */
export function followChangeAfterScroll(following: boolean, previousTop: number, metrics: FeedMetrics): FollowChange {
  if (following && metrics.scrollTop < previousTop - 0.5 && distanceFromBottom(metrics) > 1) return "unfollow";
  if (!following && metrics.scrollTop > previousTop + 0.5 && distanceFromBottom(metrics) <= RESUME_FOLLOW_PX) {
    return "follow";
  }
  return null;
}

/** Keys that scroll a focused feed back towards older lines. */
export function scrollsBack(key: string): boolean {
  return key === "ArrowUp" || key === "PageUp" || key === "Home";
}

export interface RowBox {
  id: string;
  top: number;
  bottom: number;
}

/** The reader's place: a row and how far its top sits from the feed's top edge. */
export interface FeedAnchor {
  id: string;
  offset: number;
}

/** The first row still showing below the feed's top edge. Rows come in document order, so this stops early. */
export function pickAnchor(rows: Iterable<RowBox>, viewportTop: number): FeedAnchor | null {
  for (const row of rows) {
    if (row.bottom > viewportTop) return { id: row.id, offset: row.top - viewportTop };
  }
  return null;
}

/** How far to scroll so the anchored row is back where the reader left it. */
export function anchorCorrection(anchor: FeedAnchor, rowTop: number, viewportTop: number): number {
  return rowTop - viewportTop - anchor.offset;
}
