import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "rag-starter — ask your documents",
  description: "FastAPI + Next.js retrieval-augmented generation starter with cited answers.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
