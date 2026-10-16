import { useLocation } from "./router";
import { Library } from "./screens/Library";
import { MovieScreen } from "./screens/Movie";

export function App() {
  const { path } = useLocation();
  const m = path.match(/^\/m\/([a-z0-9]+)\/?$/);
  if (m) return <MovieScreen key={m[1]} id={m[1]} />;
  return <Library />;
}
