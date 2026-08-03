import { lazy, Suspense, useEffect, type ReactNode } from 'react';
import {
  BrowserRouter,
  Navigate,
  Outlet,
  Route,
  Routes,
  useLocation,
  useNavigate,
} from 'react-router-dom';
import { isAuthed } from './lib/api';
import { AppStateProvider } from './state/AppState';
import Layout from './components/Layout';
// Login + Dashboard stay STATIC: they are the first paint on every cold load,
// so lazying them would only add a round trip to the critical path.
import Login from './pages/Login';
import Dashboard from './pages/Dashboard';

// Everything else is route-split. Measured on this tree: the app chunk goes
// from 186,991 B raw / 54,902 B gzip to 44,110 B / 14,600 B — ~143 KB raw of
// parse-and-execute removed from the cold path to the camera grid, which is
// the screen someone opens when they want to see what just triggered an alert.
const CameraDetail = lazy(() => import('./pages/CameraDetail'));
const Events = lazy(() => import('./pages/Events'));
const EventDetail = lazy(() => import('./pages/EventDetail'));
const Timeline = lazy(() => import('./pages/Timeline'));
const Settings = lazy(() => import('./pages/Settings'));
const TvPage = lazy(() => import('./pages/TvPage'));

function RequireAuth({ children }: { children: ReactNode }) {
  const location = useLocation();
  if (!isAuthed()) {
    const next = encodeURIComponent(location.pathname + location.search);
    return <Navigate to={`/login?next=${next}`} replace />;
  }
  return <>{children}</>;
}

/** Handles navigation messages posted by the service worker on notificationclick. */
function SwNavigator() {
  const navigate = useNavigate();
  useEffect(() => {
    if (!('serviceWorker' in navigator)) return;
    const onMessage = (ev: MessageEvent) => {
      const data = ev.data as { type?: string; url?: string } | null;
      if (data?.type === 'navigate' && typeof data.url === 'string') {
        try {
          const u = new URL(data.url, window.location.origin);
          navigate(u.pathname + u.search, { replace: false });
        } catch {
          /* malformed url — ignore */
        }
      }
    };
    navigator.serviceWorker.addEventListener('message', onMessage);
    return () => navigator.serviceWorker.removeEventListener('message', onMessage);
  }, [navigate]);
  return null;
}

export default function App() {
  return (
    <BrowserRouter>
      <SwNavigator />
      {/* One boundary around all routes. `null` rather than a spinner: the
          split chunks are served from the same origin and are already
          cache-warm on any repeat visit, so a flash of spinner on navigation
          would be more noticeable than the load it covers. */}
      <Suspense fallback={null}>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route
          element={
            <RequireAuth>
              <AppStateProvider>
                <Outlet />
              </AppStateProvider>
            </RequireAuth>
          }
        >
          {/* Chrome-free TV wall (kiosk deep link) — shares auth + app state. */}
          <Route path="/tv" element={<TvPage />} />
          <Route element={<Layout />}>
            <Route path="/" element={<Dashboard />} />
            <Route path="/cameras/:name" element={<CameraDetail />} />
            <Route path="/events" element={<Events />} />
            <Route path="/events/:id" element={<EventDetail />} />
            <Route path="/timeline" element={<Timeline />} />
            <Route path="/settings" element={<Settings />} />
            <Route path="/settings/:tab" element={<Settings />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Route>
      </Routes>
      </Suspense>
    </BrowserRouter>
  );
}
