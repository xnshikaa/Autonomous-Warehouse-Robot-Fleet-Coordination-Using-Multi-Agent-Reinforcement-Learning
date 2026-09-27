/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        industrial: {
          900: '#0B0F17',
          800: '#141B2D',
          700: '#1E293B',
          600: '#334155',
          500: '#475569',
          accent: '#38BDF8', // Crisp industrial cyan/blue
          success: '#10B981', // Operational green
          warning: '#F59E0B', // Safety amber
          danger: '#EF4444', // Collision red
        }
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
