/** @type {import('tailwindcss').Config} */
import typography from '@tailwindcss/typography';

export default {
  content: ['./index.html', './src/**/*.{ts,tsx,js,jsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['DM Sans', 'ui-sans-serif', 'system-ui', 'sans-serif'],
      },
      colors: {
        app: {
          bg: '#ffffff',
          panel: '#ffffff',
          border: '#e8e6e0',
          text: '#1a1a18',
          muted: '#5a5850',
          soft: '#f5f4f1',
        },
      },
      spacing: {
        appbar: '56px',
        panelLeft: '300px',
        panelRight: '310px',
        panelCollapsed: '48px',
      },
    },
  },
  plugins: [typography],
}

