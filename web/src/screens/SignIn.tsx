// Shown when the server has FH_TOKEN set and this browser hasn't signed in.
import { useState } from "react";
import { ApiError, api } from "../api";
import { useAnnouncer } from "../components/Announcer";

export function SignIn({ onSignedIn }: { onSignedIn: () => void }) {
  const [token, setToken] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const { announce } = useAnnouncer();

  return (
    <main className="page sign-in">
      <h1 className="screen-title">FrameHound</h1>
      <p className="measure">This FrameHound needs its access token. It's FH_TOKEN in the server's .env.</p>
      <form
        className="add"
        onSubmit={async (e) => {
          e.preventDefault();
          setBusy(true);
          setError(null);
          try {
            await api.signIn(token.trim());
            announce("Signed in");
            onSignedIn();
          } catch (err) {
            setError(err as ApiError);
            announce((err as ApiError).message, true);
          } finally {
            setBusy(false);
          }
        }}
      >
        <label htmlFor="token">Access token</label>
        <div className="add-row">
          <input
            id="token"
            className="field"
            type="password"
            autoComplete="current-password"
            autoCapitalize="off"
            autoCorrect="off"
            spellCheck={false}
            enterKeyHint="go"
            value={token}
            onChange={(e) => {
              setToken(e.target.value);
              setError(null);
            }}
            aria-invalid={!!error}
            aria-describedby={error ? "token-error" : undefined}
            autoFocus
          />
          <button className="btn btn-primary" type="submit" disabled={!token.trim() || busy}>
            {busy ? "Signing in…" : "Sign in"}
          </button>
        </div>
        {error && (
          <p id="token-error" className="message danger" role="alert">
            {error.message}
          </p>
        )}
      </form>
    </main>
  );
}
