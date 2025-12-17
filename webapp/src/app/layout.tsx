import "@mantine/core/styles.css";
import "@mantine/notifications/styles.css";

import "./globals.css";
import { Providers } from "./providers";
import { Metadata } from "next";

export const metadata: Metadata = {
  title: "HR Assistant | DSPy-Enhanced Chatbot",
  description: "Intelligent HR FAQ chatbot with DSPy optimization - Ask about policies, benefits, and more",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
