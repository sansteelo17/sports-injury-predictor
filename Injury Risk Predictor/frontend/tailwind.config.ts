import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // InjuryWatch brand colors - black/dark with pastel green
        'brand': {
          'black': '#0a0a0a',
          'dark': '#141414',
          'gray': '#1f1f1f',
          'green': '#86efac',      // pastel green
          'green-dark': '#4ade80', // slightly darker green for accents
          'green-light': '#bbf7d0', // lighter green
        },
        // Risk levels
        'risk-high': '#f87171',    // softer red
        'risk-medium': '#fbbf24',  // amber
        'risk-low': '#86efac',     // pastel green
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      backgroundColor: {
        'dark-primary': '#0a0a0a',
        'dark-secondary': '#141414',
        'dark-tertiary': '#1f1f1f',
      },
    },
  },
  plugins: [],
}
export default config
