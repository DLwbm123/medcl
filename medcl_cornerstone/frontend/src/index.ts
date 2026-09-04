import type { FrontendRenderer } from "@streamlit/component-v2-lib";
import { CleanupBag } from "./lifecycle";
import { parseEnvelope } from "./volumeProtocol";
import { mountViewer, ViewerState } from "./viewer";

const render: FrontendRenderer<ViewerState, Uint8Array | ArrayBuffer> = (args) => {
  const bag = new CleanupBag();
  let cancelled = false;
  try {
    const envelope = parseEnvelope(args.data);
    void mountViewer(args, envelope, bag, () => cancelled).catch((error: unknown) => {
      if (cancelled) return;
      bag.close();
      const code = error instanceof Error && error.message === "webgl_unavailable" ? "WEBGL_UNAVAILABLE" : "VIEWER_INIT_FAILED";
      const notice = document.createElement("div");
      notice.className = "medcl-error";
      notice.setAttribute("role", "alert");
      notice.textContent = code === "WEBGL_UNAVAILABLE"
        ? "WebGL2 is unavailable. Use the static slice fallback below."
        : "The 3D viewer could not start. Use the static slice fallback below.";
      args.parentElement.replaceChildren(notice);
      args.setStateValue("viewer_ready", false);
      args.setStateValue("viewer_error_code", code);
    });
  } catch {
    const notice = document.createElement("div");
    notice.className = "medcl-error";
    notice.setAttribute("role", "alert");
    notice.textContent = "The preview payload was rejected before rendering.";
    args.parentElement.replaceChildren(notice);
    args.setStateValue("viewer_ready", false);
    args.setStateValue("viewer_error_code", "INVALID_ENVELOPE");
  }
  return () => {
    cancelled = true;
    bag.close();
    args.parentElement.replaceChildren();
  };
};

export default render;
