import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // Premier League inspired palette
        'pl-purple': '#38003c',
        'pl-green': '#00ff85',
        'pl-pink': '#e90052',
        // Risk levels
        'risk-high': '#dc2626',
        'risk-medium': '#f59e0b',
        'risk-low': '#16a34a',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
export default config
