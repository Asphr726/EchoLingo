import { describe, expect, it } from "vitest";
import { App } from "./App";

describe("desktop foundation", () => {
  it("renders the application shell", () => {
    const view = App();
    expect(view.type).toBe("main");
    expect(view.props.className).toBe("shell");
  });
});
