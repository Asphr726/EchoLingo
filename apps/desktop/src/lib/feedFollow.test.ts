import { describe, expect, it } from "vitest";
import {
  anchorCorrection,
  distanceFromBottom,
  followChangeAfterScroll,
  pickAnchor,
  RESUME_FOLLOW_PX,
  scrollsBack,
  type RowBox,
} from "./feedFollow";

const metrics = (scrollTop: number) => ({ scrollTop, scrollHeight: 2000, clientHeight: 500 });

describe("followChangeAfterScroll", () => {
  it("stops following as soon as the reader scrolls up, however little", () => {
    expect(followChangeAfterScroll(true, 1500, metrics(1497))).toBe("unfollow");
  });

  it("keeps following when the browser clamps the position after the content shrank", () => {
    // 2000 px of content shrank to 1900 px: the bottom moved from 1500 to 1400.
    expect(followChangeAfterScroll(true, 1500, { scrollTop: 1400, scrollHeight: 1900, clientHeight: 500 })).toBeNull();
  });

  it("keeps following while the feed only moves down", () => {
    expect(followChangeAfterScroll(true, 1200, metrics(1500))).toBeNull();
  });

  it("resumes following when the reader scrolls down to the newest lines", () => {
    expect(followChangeAfterScroll(false, 1300, metrics(1500 - RESUME_FOLLOW_PX))).toBe("follow");
  });

  it("does not resume while the reader is still above the newest lines", () => {
    expect(followChangeAfterScroll(false, 900, metrics(1000))).toBeNull();
  });

  it("does not resume on an upward scroll that ends near the bottom", () => {
    expect(followChangeAfterScroll(false, 1500, metrics(1480))).toBeNull();
  });
});

describe("pickAnchor", () => {
  const rows: RowBox[] = [
    { id: "a", top: -180, bottom: -60 },
    { id: "b", top: -60, bottom: 40 },
    { id: "c", top: 40, bottom: 160 },
  ];

  it("anchors the first row still showing below the top edge", () => {
    expect(pickAnchor(rows, 0)).toEqual({ id: "b", offset: -60 });
  });

  it("stops reading rows once it has found one", () => {
    let read = 0;
    function* lazyRows() {
      for (const row of rows) {
        read += 1;
        yield row;
      }
    }
    pickAnchor(lazyRows(), 0);
    expect(read).toBe(2);
  });

  it("returns null when every row is above the viewport", () => {
    expect(pickAnchor(rows, 500)).toBeNull();
  });
});

describe("anchorCorrection", () => {
  it("scrolls back up by the height of a row evicted from the top", () => {
    // The anchored row sat 20 px below the edge; evicting a 100 px row above it
    // moved it to -80 px, so the feed must scroll up by 100 px.
    expect(anchorCorrection({ id: "b", offset: 20 }, -80, 0)).toBe(-100);
  });

  it("scrolls down when a row above the anchor grows", () => {
    expect(anchorCorrection({ id: "b", offset: 20 }, 56, 0)).toBe(36);
  });

  it("measures against the feed's top edge, wherever the feed sits on the page", () => {
    expect(anchorCorrection({ id: "b", offset: 20 }, 320, 300)).toBe(0);
  });
});

describe("helpers", () => {
  it("never reports a negative distance from the bottom", () => {
    expect(distanceFromBottom({ scrollTop: 1600, scrollHeight: 2000, clientHeight: 500 })).toBe(0);
  });

  it("treats only backward navigation keys as scrolling back", () => {
    expect(["ArrowUp", "PageUp", "Home"].every(scrollsBack)).toBe(true);
    expect(["ArrowDown", "PageDown", "End", " "].some(scrollsBack)).toBe(false);
  });
});
