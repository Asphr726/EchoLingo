import { CaptionWindow } from "./components/CaptionWindow";
import { MainWindow } from "./components/MainWindow";

export function App() {
  const windowName =
    typeof window === "undefined"
      ? null
      : new URLSearchParams(window.location.search).get("window");
  return windowName === "caption" ? <CaptionWindow /> : <MainWindow />;
}
