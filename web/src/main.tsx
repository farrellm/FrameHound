import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { AnnouncerProvider } from "./components/Announcer";
import "./styles.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <AnnouncerProvider>
      <App />
    </AnnouncerProvider>
  </StrictMode>,
);
