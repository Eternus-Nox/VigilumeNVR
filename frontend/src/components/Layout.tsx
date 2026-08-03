/**
 * Authenticated app shell: header with connection state, adaptive nav
 * (bottom tab bar on mobile, left rail on desktop), toast stack, and the
 * unseen-events badge (cleared when the Events section is viewed).
 */
import { useEffect } from 'react';
import { NavLink, Outlet, useLocation, useNavigate } from 'react-router-dom';
import { clearToken } from '../lib/api';
import { useAppState, useUiLive } from '../state/AppState';
import Toasts from './Toasts';

const NAV_CAMERAS = {
  to: '/',
  label: 'Cameras',
  icon: (
    <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <rect x="2.5" y="6" width="13" height="11" rx="2" />
      <path d="M15.5 10.5 21 7.5v9l-5.5-3" />
    </svg>
  ),
};

const NAV_EVENTS = {
  to: '/events',
  label: 'Events',
  icon: (
    <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <circle cx="12" cy="12" r="8.5" />
      <path d="M12 7.5V12l3 2.2" />
    </svg>
  ),
};

const NAV_TIMELINE = {
  to: '/timeline',
  label: 'Timeline',
  icon: (
    <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M3 12h18" />
      <path d="M6 8.5v7M12 6v12M18 8.5v7" />
    </svg>
  ),
};

// Admin: full Settings. Both point at /settings; the Settings page resolves the
// correct default tab per role, so the viewer's entry lands on Groups.
const NAV_SETTINGS = {
  to: '/settings',
  label: 'Settings',
  icon: (
    <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.7 1.7 0 0 0 .34 1.87l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.7 1.7 0 0 0-1.87-.34 1.7 1.7 0 0 0-1 1.55V21a2 2 0 1 1-4 0v-.09a1.7 1.7 0 0 0-1-1.55 1.7 1.7 0 0 0-1.87.34l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.7 1.7 0 0 0 .34-1.87 1.7 1.7 0 0 0-1.55-1H3a2 2 0 1 1 0-4h.09a1.7 1.7 0 0 0 1.55-1 1.7 1.7 0 0 0-.34-1.87l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.7 1.7 0 0 0 1.87.34h.01a1.7 1.7 0 0 0 1-1.55V3a2 2 0 1 1 4 0v.09a1.7 1.7 0 0 0 1 1.55h.01a1.7 1.7 0 0 0 1.87-.34l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.7 1.7 0 0 0-.34 1.87v.01a1.7 1.7 0 0 0 1.55 1H21a2 2 0 1 1 0 4h-.09a1.7 1.7 0 0 0-1.55 1z" />
    </svg>
  ),
};

// Viewer: the only Settings capability a viewer has is shared camera groups
// (+ a per-device push toggle), so the entry is labelled "Groups".
const NAV_GROUPS = {
  to: '/settings',
  label: 'Groups',
  icon: (
    <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <rect x="3" y="3" width="8" height="8" rx="1.5" />
      <rect x="13" y="3" width="8" height="8" rx="1.5" />
      <rect x="3" y="13" width="8" height="8" rx="1.5" />
      <rect x="13" y="13" width="8" height="8" rx="1.5" />
    </svg>
  ),
};

export default function Layout() {
  // Layout subscribes to the hot slice for the conn dot + unseen badge; its
  // children come from the router's <Outlet/> (reference-stable elements), so
  // this re-render does not cascade into the page below.
  const { clearUnseen, isAdmin } = useAppState();
  const { socketStatus, unseenEvents } = useUiLive();
  const location = useLocation();
  const navigate = useNavigate();

  const nav = [
    NAV_CAMERAS,
    NAV_EVENTS,
    NAV_TIMELINE,
    isAdmin ? NAV_SETTINGS : NAV_GROUPS,
  ];

  // Viewing any /events route clears the badge.
  useEffect(() => {
    if (location.pathname.startsWith('/events')) clearUnseen();
  }, [location.pathname, clearUnseen]);

  const logout = () => {
    clearToken();
    navigate('/login', { replace: true });
  };

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand">
          <img src="/favicon.svg" alt="" width="26" height="26" />
          <span>Vigilume</span>
        </div>
        <div className="topbar-right">
          <span
            className={`conn-dot conn-${socketStatus}`}
            title={`live feed: ${socketStatus}`}
            aria-label={`live connection ${socketStatus}`}
          />
          <button type="button" className="btn btn-ghost btn-sm" onClick={logout}>
            Sign out
          </button>
        </div>
      </header>

      <nav className="nav" aria-label="Primary">
        {nav.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === '/'}
            className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
          >
            <span className="nav-icon">
              {item.icon}
              {item.to === '/events' && unseenEvents > 0 && (
                <span className="badge">{unseenEvents > 99 ? '99+' : unseenEvents}</span>
              )}
            </span>
            <span className="nav-label">{item.label}</span>
          </NavLink>
        ))}
      </nav>

      <main className="content">
        <Outlet />
      </main>

      <Toasts />
    </div>
  );
}
