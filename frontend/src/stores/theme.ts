import { create } from "zustand";

export type Theme = "dark" | "light";

const KEY = "mmde-theme";

function initialTheme(): Theme {
  try {
    const v = localStorage.getItem(KEY);
    if (v === "light" || v === "dark") return v;
  } catch {
    /* ignore */
  }
  return "dark";
}

function apply(theme: Theme) {
  document.documentElement.classList.toggle("dark", theme === "dark");
}

interface ThemeState {
  theme: Theme;
  toggle: () => void;
}

export const useThemeStore = create<ThemeState>((set, get) => ({
  theme: initialTheme(),
  toggle: () => {
    const theme: Theme = get().theme === "dark" ? "light" : "dark";
    try {
      localStorage.setItem(KEY, theme);
    } catch {
      /* ignore */
    }
    apply(theme);
    set({ theme });
  },
}));

apply(initialTheme());
