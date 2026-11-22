import { useEffect, useRef, useState } from "react";

export interface HandSceneProps {
  gesture: string | null;
  confidence: number;
  activation: number;
}

/**
 * Thin adapter around the bionic prosthetic hand rig defined in
 * assets/hand.html (ported unmodified to frontend/public/hand.html).
 *
 * The rig is a large hand-tuned Three.js scene (procedural geometry,
 * FK-verified pinch/tripod contact, studio lighting). Re-implementing it
 * in react-three-fiber would risk drifting from that visual, so it is
 * embedded in a sandboxed <iframe> and driven via postMessage against its
 * window.HandRig.{setPose,setConfidence,setActivation} contract. Callers
 * of HandScene don't need to know it's iframe-based, this is the only
 * file that would change if the rig were ever ported natively.
 */
export function HandScene({ gesture, confidence, activation }: HandSceneProps) {
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    function onMessage(evt: MessageEvent) {
      const data = evt.data;
      if (data && data.source === "myocopilot-hand" && data.type === "handrig-ready") {
        setReady(true);
      }
    }
    window.addEventListener("message", onMessage);
    return () => window.removeEventListener("message", onMessage);
  }, []);

  function call(method: string, ...args: unknown[]) {
    const win = iframeRef.current?.contentWindow;
    if (!win) return;
    win.postMessage(
      { source: "myocopilot-hand", type: "handrig-call", method, args },
      "*",
    );
  }

  useEffect(() => {
    if (!ready) return;
    if (gesture) call("setPose", gesture);
    call("setConfidence", confidence);
    call("setActivation", activation);
  }, [ready, gesture, confidence, activation]);

  return (
    <div className="hand-scene">
      <iframe
        ref={iframeRef}
        title="MyoCopilot prosthetic hand"
        src="/hand.html"
        sandbox="allow-scripts allow-same-origin"
        className="hand-scene__frame"
      />
    </div>
  );
}
