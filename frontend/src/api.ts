import axios from 'axios';
import { getOrCreateSessionId } from './lib/session';

export const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8082',
});

// Inject X-Session-Id on every request so Sentry traces and MLOps
// prediction-log rows are tagged with the same id (cross-tier loop).
// See frontend/src/lib/session.ts for the lifecycle.
api.interceptors.request.use((config) => {
  config.headers = config.headers ?? {};
  config.headers['X-Session-Id'] = getOrCreateSessionId();
  return config;
});

/** Re-export so components that need the raw id (e.g. for display) can grab it. */
export { getOrCreateSessionId } from './lib/session';
