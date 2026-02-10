import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'EPL Injury Risk Predictor',
  description: 'ML-powered injury risk predictions for Premier League players',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
