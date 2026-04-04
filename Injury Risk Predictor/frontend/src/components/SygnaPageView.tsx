"use client";

import { useEffect, useRef } from "react";
import { usePathname } from "next/navigation";
import { trackYaraPageView } from "@/analytics/sygna";

export function SygnaPageView() {
  const pathname = usePathname();
  const lastTrackedPath = useRef("");

  useEffect(() => {
    const query =
      typeof window !== "undefined" ? window.location.search || "" : "";
    const nextPath = `${pathname || "/"}${query}`;

    if (!nextPath || nextPath === lastTrackedPath.current) {
      return;
    }

    lastTrackedPath.current = nextPath;
    trackYaraPageView();
  }, [pathname]);

  return null;
}
