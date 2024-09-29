"use client";

import { useTheme } from "next-themes";
import { useEffect, useState } from "react";
import { GithubIcon } from "lucide-react";

export function GithubButton() {
  const [mounted, setMounted] = useState(false);

  // Wait until the theme is mounted to avoid mismatch between client and server rendering
  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) return null;

  return (
    <a
      href="https://github.com/smugg99/hackathon-2024"
      target="_blank" // Open in a new tab
      rel="noopener noreferrer" // Security measure
    >
      <button
        className="p-2 rounded-full hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
        aria-label="GitHub"
      >
        <GithubIcon className="w-6 h-6 dark:text-gray-100" />
      </button>
    </a>
  );
}
