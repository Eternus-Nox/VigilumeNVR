/**
 * "Where is that setting?" — a search box over the whole settings surface.
 *
 * Settings live on nine tabs and about fifty cards, and the word someone
 * reaches for is usually the SYMPTOM, not the feature: "parked car", "too many
 * notifications", "clips start too late". `searchIndex.ts` carries those words
 * deliberately, which is why this searches an index rather than the rendered
 * page — filtering the DOM would only match text already on screen, which is
 * exactly what someone who cannot find the setting has not got in front of
 * them.
 *
 * Picking a result switches tab and then scrolls to the card. The scroll is
 * done by matching the card's <h2> AFTER the tab has rendered, so it needs no
 * ids threaded through every card; if a heading is renamed and the index is
 * not, the result degrades to "lands on the right tab" rather than breaking.
 */
import { useEffect, useRef, useState } from 'react';
import { searchSettings, type SettingsEntry } from '../pages/settings/searchIndex';

interface Props {
  isAdmin: boolean;
  /** Switch tabs. The scroll is queued for after the new tab paints. */
  onGo: (tab: string) => void;
}

/** Scroll to the card with this exact heading, and flash it so the eye lands.
 *
 * Two markups to match, because settings grew two card styles: a plain
 * `<section class="card">` with an `<h2>`, and `SettingsDisclosure`, whose
 * title is a `<span class="settings-disclosure-title">` inside a `<summary>`.
 * Matching only the first silently did nothing on Notifications, Home
 * Assistant and the System cards — the search would switch tab and then
 * appear to have missed. */
function revealCard(heading: string): boolean {
  const candidates = document.querySelectorAll(
    '.settings-section h2, .settings-disclosure-title, .privacy-summary-title',
  );
  const match = Array.from(candidates).find(
    (h) => h.textContent?.trim() === heading,
  );
  if (!match) return false;
  // A closed <details> would scroll to a collapsed card showing nothing.
  const details = match.closest('details');
  if (details && !details.open) details.open = true;
  const card = match.closest('.card') ?? match;
  card.scrollIntoView({ behavior: 'smooth', block: 'center' });
  card.classList.add('card-found');
  window.setTimeout(() => card.classList.remove('card-found'), 1600);
  return true;
}

export default function SettingsSearch({ isAdmin, onGo }: Props) {
  const [query, setQuery] = useState('');
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  // The card to reveal once the destination tab has painted. A tab switch is a
  // route change and a re-render, so the heading does not exist yet at click
  // time — scrolling immediately would silently find nothing.
  const pending = useRef<string | null>(null);
  const boxRef = useRef<HTMLDivElement | null>(null);

  const results = searchSettings(query, { isAdmin });

  useEffect(() => {
    if (!pending.current) return;
    const heading = pending.current;
    // Two frames: one for the tab to mount, one for layout. Retried briefly
    // because a tab that fetches (the camera list, the model list) can render
    // its cards a beat later.
    let tries = 0;
    const tick = () => {
      if (revealCard(heading) || tries++ > 20) {
        pending.current = null;
        return;
      }
      window.setTimeout(tick, 50);
    };
    const id = window.setTimeout(tick, 0);
    return () => window.clearTimeout(id);
  });

  useEffect(() => {
    const onDown = (e: MouseEvent) => {
      if (boxRef.current && !boxRef.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', onDown);
    return () => document.removeEventListener('mousedown', onDown);
  }, []);

  const go = (entry: SettingsEntry) => {
    pending.current = entry.card;
    setQuery('');
    setOpen(false);
    onGo(entry.tab);
  };

  return (
    <div className="settings-search" ref={boxRef}>
      <input
        type="search"
        value={query}
        placeholder="Search settings — try “parked car” or “too many alerts”"
        aria-label="Search settings"
        onChange={(e) => {
          setQuery(e.target.value);
          setActive(0);
          setOpen(true);
        }}
        onFocus={() => setOpen(true)}
        onKeyDown={(e) => {
          if (!results.length) return;
          if (e.key === 'ArrowDown') {
            e.preventDefault();
            setActive((i) => (i + 1) % results.length);
          } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            setActive((i) => (i - 1 + results.length) % results.length);
          } else if (e.key === 'Enter') {
            e.preventDefault();
            go(results[active]);
          } else if (e.key === 'Escape') {
            setOpen(false);
          }
        }}
      />
      {open && query.trim().length >= 2 && (
        <ul className="settings-search-results" role="listbox">
          {results.length === 0 ? (
            <li className="settings-search-empty">
              Nothing matches “{query.trim()}”.
            </li>
          ) : (
            results.map((entry, i) => (
              <li key={`${entry.tab}-${entry.card}`}>
                <button
                  type="button"
                  className={`settings-search-hit ${i === active ? 'is-active' : ''}`}
                  role="option"
                  aria-selected={i === active}
                  onMouseEnter={() => setActive(i)}
                  onClick={() => go(entry)}
                >
                  <span className="settings-search-label">{entry.label}</span>
                  <span className="settings-search-where">
                    {entry.tab === 'faces'
                      ? 'Faces & plates'
                      : entry.tab.charAt(0).toUpperCase() + entry.tab.slice(1)}{' '}
                    › {entry.card}
                  </span>
                </button>
              </li>
            ))
          )}
        </ul>
      )}
    </div>
  );
}
