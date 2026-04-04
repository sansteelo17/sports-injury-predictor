import type { Metadata, Viewport } from 'next';
import './globals.css';
import { SygnaPageView } from '@/components/SygnaPageView';

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  maximumScale: 1,
};

export const metadata: Metadata = {
  title: 'YaraSports - Injury Risk Predictor',
  description: 'ML-powered injury risk predictions for Premier League players',
  icons: {
    icon: '/icon.svg',
    shortcut: '/icon.svg',
    apple: '/icon.svg',
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="antialiased">
        <SygnaPageView />
        {children}
      </body>
    </html>
  );
}
