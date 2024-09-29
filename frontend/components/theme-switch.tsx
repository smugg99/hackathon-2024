"use client";

import { useTheme } from "next-themes";
import { useEffect, useState } from "react";
import { Sun, MoonStar } from "lucide-react";

export function ThemeSwitch() {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  // Wait until the theme is mounted to avoid mismatch between client and server rendering
  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) return null;

  return (
    <button
      className="p-2 rounded-full hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
      onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
      aria-label="Toggle theme"
    >
      {theme === "dark" ? (
        <Sun className="w-6 h-6 text-yellow-400" />
      ) : (
        <MoonStar className="w-6 h-6 text-blue-800 dark:text-gray-100" />
      )}
    </button>
  );
}
