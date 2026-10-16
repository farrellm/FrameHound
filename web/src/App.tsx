import { useEffect, useState } from "react";
import { api, onSignedOut } from "./api";
import { useLocation } from "./router";
import { Library } from "./screens/Library";
import { MovieScreen } from "./screens/Movie";
import { SignIn } from "./screens/SignIn";

export function App() {
  const { path } = useLocation();
  const [signedOut, setSignedOut] = useState(false);
  // Bumped on sign-in so the screen remounts and loads again.
  const [session, setSession] = useState(0);

  useEffect(() => onSignedOut(() => setSignedOut(true)), []);
  useEffect(() => {
    api.health().then(
      (h) => h.auth_required && !h.signed_in && setSignedOut(true),
      () => {},
    );
  }, []);

  if (signedOut)
    return (
      <SignIn
        onSignedIn={() => {
          setSignedOut(false);
          setSession((n) => n + 1);
        }}
      />
    );
  const m = path.match(/^\/m\/([a-z0-9]+)\/?$/);
  if (m) return <MovieScreen key={`${m[1]}-${session}`} id={m[1]} />;
  return <Library key={session} />;
}
