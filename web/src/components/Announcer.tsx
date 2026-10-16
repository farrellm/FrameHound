// Live regions and the toast (ui-design.md §6): polite for progress, assertive for errors.
import { createContext, useCallback, useContext, useRef, useState, type ReactNode } from "react";

interface Ctx {
  announce: (text: string, assertive?: boolean) => void;
  toast: (text: string) => void;
}

const AnnounceContext = createContext<Ctx>({ announce: () => {}, toast: () => {} });

export function AnnouncerProvider({ children }: { children: ReactNode }) {
  const [polite, setPolite] = useState("");
  const [assertive, setAssertive] = useState("");
  const [toastText, setToast] = useState<string | null>(null);
  const timer = useRef<number>(undefined);

  const announce = useCallback((text: string, isAssertive = false) => {
    const set = isAssertive ? setAssertive : setPolite;
    set("");
    requestAnimationFrame(() => set(text));
  }, []);

  const toast = useCallback(
    (text: string) => {
      setToast(text);
      announce(text);
      window.clearTimeout(timer.current);
      timer.current = window.setTimeout(() => setToast(null), 3500);
    },
    [announce],
  );

  return (
    <AnnounceContext.Provider value={{ announce, toast }}>
      {children}
      <div className="visually-hidden" aria-live="polite">
        {polite}
      </div>
      <div className="visually-hidden" aria-live="assertive">
        {assertive}
      </div>
      {toastText && (
        <div className="toast" role="status">
          {toastText}
        </div>
      )}
    </AnnounceContext.Provider>
  );
}

export const useAnnouncer = () => useContext(AnnounceContext);
