// A sheet rises from the bottom on phones and anchors right on desktop.
// Focus moves in when it opens and returns to the trigger when it closes.
import { useEffect, useRef, type ReactNode } from "react";

interface Props {
  label: string;
  onClose: () => void;
  children: ReactNode;
}

const FOCUSABLE =
  'button:not([disabled]), [href], input:not([disabled]), textarea, select, [tabindex]:not([tabindex="-1"])';

export function Sheet({ label, onClose, children }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const close = useRef(onClose);
  close.current = onClose;

  useEffect(() => {
    const trigger = document.activeElement as HTMLElement | null;
    const node = ref.current!;
    (
      node.querySelector<HTMLElement>("[data-autofocus]") ??
      node.querySelector<HTMLElement>(FOCUSABLE) ??
      node
    ).focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        close.current();
      } else if (e.key === "Tab") {
        const items = Array.from(node.querySelectorAll<HTMLElement>(FOCUSABLE));
        if (!items.length) return;
        const first = items[0];
        const last = items[items.length - 1];
        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      }
    };
    document.addEventListener("keydown", onKey, true);
    return () => {
      document.removeEventListener("keydown", onKey, true);
      trigger?.focus?.();
    };
  }, []);

  return (
    <>
      <div className="scrim" onClick={() => close.current()} />
      <div className="sheet" role="dialog" aria-modal="true" aria-label={label} ref={ref} tabIndex={-1}>
        <div className="grabber" aria-hidden="true" />
        {children}
      </div>
    </>
  );
}
