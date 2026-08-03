import { useState, type FormEvent } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { api, setRole, setToken, setUsername } from '../lib/api';

export default function Login() {
  // Default the username to the built-in admin for one-field convenience; the
  // env admin still logs in as "admin".
  const [username, setUsernameInput] = useState('admin');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const navigate = useNavigate();
  const [params] = useSearchParams();

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!username.trim() || !password || busy) return;
    setBusy(true);
    setError(null);
    try {
      const res = await api.login(username.trim(), password);
      setToken(res.token);
      // Cache role + username so the app shell knows what to gate before the
      // GET /api/auth/me confirmation lands (a legacy backend omits them).
      if (res.role) setRole(res.role);
      setUsername(res.username ?? username.trim());
      const next = params.get('next');
      navigate(next && next.startsWith('/') ? next : '/', { replace: true });
    } catch (err) {
      setError(
        err instanceof Error && err.message !== '401 Unauthorized'
          ? err.message
          : 'Incorrect username or password',
      );
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="login-page">
      <form className="login-card" onSubmit={submit}>
        <img src="/favicon.svg" alt="" width="56" height="56" />
        <h1>Vigilume NVR</h1>
        <p className="muted">Sign in to your security console</p>
        <input
          type="text"
          autoComplete="username"
          placeholder="Username"
          value={username}
          onChange={(e) => setUsernameInput(e.target.value)}
          aria-label="Username"
        />
        <input
          type="password"
          autoFocus
          autoComplete="current-password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          aria-label="Password"
        />
        {error && <p className="form-error">{error}</p>}
        <button
          type="submit"
          className="btn btn-primary btn-block"
          disabled={busy || !password || !username.trim()}
        >
          {busy ? 'Signing in…' : 'Sign in'}
        </button>
      </form>
      <p className="muted login-legal">
        <a href="/privacy.html" target="_blank" rel="noopener">
          Privacy Policy
        </a>
        {' · '}
        <a href="/terms.html" target="_blank" rel="noopener">
          Terms of Use
        </a>
      </p>
    </div>
  );
}
