# Frontend Dashboard

React + TypeScript dashboard for `inboxpilot-lite` backend.

## Stack
- Vite
- React + TypeScript
- TailwindCSS
- Recharts

## Prerequisites
- Node.js 18+
- Backend running at `http://localhost:8000`

## Run
```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## Build
```bash
npm run build
npm run preview
```

## Features
- Top bar with backend status, active classifier/model, dark mode toggle
- Single classify and batch classify workflows
- Live stats cards and charts
- Recent activity table with filters and detail drawer
- Polling every 5 seconds with pause toggle
