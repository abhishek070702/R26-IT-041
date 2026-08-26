import { useEffect, useState } from "react";
import {
  CONNECTION_LABELS,
  FALLBACK_API_PAYLOAD,
  LIVE_STATUS_URL,
  mapApiToViewModel,
} from "../liveDemoStatus";

export function useLiveStatus() {
  const [connectionState, setConnectionState] = useState("waiting");
  const [viewModel, setViewModel] = useState(() =>
    mapApiToViewModel(FALLBACK_API_PAYLOAD, "waiting"),
  );

  useEffect(() => {
    let cancelled = false;

    async function fetchStatus() {
      try {
        const response = await fetch(LIVE_STATUS_URL, {
          cache: "no-store",
        });

        if (!response.ok) {
          throw new Error(`Live status request failed (${response.status})`);
        }

        const payload = await response.json();
        if (cancelled) {
          return;
        }

        setConnectionState("connected");
        setViewModel(mapApiToViewModel(payload, "connected"));
      } catch {
        if (cancelled) {
          return;
        }

        setConnectionState("fallback");
        setViewModel(mapApiToViewModel(FALLBACK_API_PAYLOAD, "fallback"));
      }
    }

    fetchStatus();
    const intervalId = window.setInterval(fetchStatus, 1000);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, []);

  return {
    connectionState,
    connectionLabel: CONNECTION_LABELS[connectionState],
    ...viewModel,
  };
}
