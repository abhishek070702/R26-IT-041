/** Backend origin for all API calls. Override with `VITE_API_BASE_URL` in `frontend/.env`. */
export const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL as string | undefined
)?.trim() || "http://127.0.0.1:8000";
