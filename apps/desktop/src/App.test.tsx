import { describe, expect, it } from "vitest";
import { App } from "./App";

describe("desktop window routing", () => {
  it("uses the main product window by default", () => {
    const view = App();
    expect(view.type.name).toBe("MainWindow");
  });
});
