import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "HR Assistant | DSPy-Enhanced Chatbot",
  description: "Intelligent HR FAQ chatbot with DSPy optimization - Ask about policies, benefits, and more",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
