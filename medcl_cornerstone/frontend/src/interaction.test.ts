import { describe, expect, it } from "vitest";
import { mprBindingPlan, MprPrimaryTool } from "./interaction";

const primaryTools = (selected: MprPrimaryTool) => Object.entries(mprBindingPlan(selected))
  .filter(([, bindings]) => bindings.includes("Primary"))
  .map(([tool]) => tool);

describe("MPR primary tool bindings", () => {
  it.each([
    ["windowLevel", "windowLevel"],
    ["pan", "pan"],
    ["zoom", "zoom"],
    ["crosshairs", "crosshairs"],
  ] as const)("keeps only %s on Primary", (selected, expected) => {
    expect(primaryTools(selected)).toEqual([expected]);
  });

  it("preserves middle Pan, right Zoom, and wheel Scroll", () => {
    for (const selected of ["windowLevel", "pan", "zoom", "crosshairs"] as const) {
      const plan = mprBindingPlan(selected);
      expect(plan.pan).toContain("Auxiliary");
      expect(plan.zoom).toContain("Secondary");
      expect(plan.stackScroll).toEqual(["Wheel"]);
      for (const bindings of Object.values(plan)) expect(new Set(bindings).size).toBe(bindings.length);
    }
  });

  it("is idempotent for repeated selection", () => {
    expect(mprBindingPlan("pan")).toEqual(mprBindingPlan("pan"));
  });
});
