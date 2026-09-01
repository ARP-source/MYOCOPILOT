# MyoCopilot frontend dev server (Vite).
$ErrorActionPreference = "Stop"
Push-Location frontend
try {
    npm run dev
} finally {
    Pop-Location
}
