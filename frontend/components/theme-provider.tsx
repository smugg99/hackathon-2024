"use client";

import * as React from "react";
import { ThemeProvider as NextThemesProvider } from "next-themes";
import { type ThemeProviderProps } from "next-themes/dist/types";

export function ThemeProvider({ children, ...props }: ThemeProviderProps) {
  return (
    <NextThemesProvider
      attribute="class"      // This adds the theme class (e.g., "light" or "dark") to the <html> element
      defaultTheme="dark"    // Set the default theme to dark
      enableSystem={true}    // Enable switching based on system theme
      disableTransitionOnChange={true} // Prevents theme transition flickering
      {...props}
    >
      {children}
    </NextThemesProvider>
  );
}
