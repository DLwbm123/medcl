export type MprTool = "windowLevel" | "pan" | "zoom" | "crosshairs" | "stackScroll";
export type MprPrimaryTool = Exclude<MprTool, "stackScroll">;
export type MouseBinding = "Primary" | "Auxiliary" | "Secondary" | "Wheel";

export const mprBindingPlan = (primary: MprPrimaryTool): Record<MprTool, MouseBinding[]> => ({
  windowLevel: primary === "windowLevel" ? ["Primary"] : [],
  pan: [...(primary === "pan" ? ["Primary"] as MouseBinding[] : []), "Auxiliary"],
  zoom: [...(primary === "zoom" ? ["Primary"] as MouseBinding[] : []), "Secondary"],
  crosshairs: primary === "crosshairs" ? ["Primary"] : [],
  stackScroll: ["Wheel"],
});
