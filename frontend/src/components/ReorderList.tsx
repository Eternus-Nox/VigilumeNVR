/**
 * Dependency-free reorderable list. Every row gets a drag handle (pointer
 * events + CSS transforms — DOM order never changes mid-drag, so pointer
 * capture and touch streams stay stable) plus up/down arrow buttons as the
 * accessible / small-screen fallback. The handle sets `touch-action: none`
 * (CSS) so touch drags don't scroll the page. The new order is committed
 * exactly once, on drop or arrow press — never mid-drag.
 */
import {
  useEffect,
  useRef,
  useState,
  type CSSProperties,
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
} from 'react';

interface DragState {
  key: string;
  /** Index the dragged row started at. */
  from: number;
  /** Current insertion index (splice semantics after removing `from`). */
  to: number;
  /** Pointer travel in px, applied to the dragged row. */
  dy: number;
  /** Vacated-slot size: dragged row height + list gap. */
  stride: number;
}

interface ReorderListProps<T> {
  items: T[];
  itemKey: (item: T) => string;
  /** Full reordered list; fired once per drop / arrow press. */
  onCommit: (next: T[]) => void;
  /** Row content, rendered between the drag handle and the arrows. */
  renderItem: (item: T) => ReactNode;
  /** Extra class for each row `<li>` (row layout/skin comes from it). */
  itemClassName?: string;
  ariaLabel?: string;
}

export default function ReorderList<T>({
  items,
  itemKey,
  onCommit,
  renderItem,
  itemClassName,
  ariaLabel,
}: ReorderListProps<T>) {
  const listRef = useRef<HTMLUListElement | null>(null);
  const [drag, setDrag] = useState<DragState | null>(null);
  const cleanupRef = useRef<(() => void) | null>(null);

  // Drop the window listeners if we unmount mid-drag.
  useEffect(() => () => cleanupRef.current?.(), []);

  /** Splice-move: remove `from`, insert at `to` (index after the removal). */
  const move = (from: number, to: number) => {
    if (to < 0 || to >= items.length || to === from) return;
    const next = [...items];
    const [it] = next.splice(from, 1);
    next.splice(to, 0, it);
    onCommit(next);
  };

  const startDrag = (e: ReactPointerEvent<HTMLButtonElement>, key: string, from: number) => {
    if (items.length < 2 || cleanupRef.current) return;
    // Primary button / single touch only.
    if (e.button !== 0) return;
    e.preventDefault();
    const list = listRef.current;
    if (!list) return;
    const rows = Array.from(list.querySelectorAll<HTMLElement>(':scope > li'));
    if (rows.length !== items.length) return;
    const rects = rows.map((r) => r.getBoundingClientRect());
    const gap = rects.length > 1 ? Math.max(0, rects[1].top - rects[0].bottom) : 0;
    const stride = rects[from].height + gap;
    const centers = rects.map((r) => r.top + r.height / 2);
    const startY = e.clientY;
    try {
      e.currentTarget.setPointerCapture(e.pointerId);
    } catch {
      /* capture unsupported — window listeners still track the pointer */
    }

    // Mutable session so the listeners never see stale state.
    const session = { from, to: from };
    setDrag({ key, from, to: from, dy: 0, stride });

    const onMove = (ev: PointerEvent) => {
      const dy = ev.clientY - startY;
      const midY = centers[from] + dy;
      // Insertion index among the other rows = how many of their static
      // centers the dragged row's (moving) center has passed.
      let to = 0;
      for (let j = 0; j < centers.length; j += 1) {
        if (j !== from && centers[j] < midY) to += 1;
      }
      session.to = to;
      setDrag({ key, from, to, dy, stride });
    };
    const finish = (commit: boolean) => () => {
      cleanupRef.current?.();
      cleanupRef.current = null;
      setDrag(null);
      if (commit && session.to !== session.from) move(session.from, session.to);
    };
    const onUp = finish(true);
    const onCancel = finish(false);
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    window.addEventListener('pointercancel', onCancel);
    cleanupRef.current = () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
      window.removeEventListener('pointercancel', onCancel);
    };
  };

  /** Visual offset for row `index` while a drag is in flight. */
  const rowStyle = (index: number): CSSProperties | undefined => {
    if (!drag) return undefined;
    if (index === drag.from) return { transform: `translateY(${drag.dy}px)` };
    // Position among the other rows once the dragged one is removed:
    const without = index > drag.from ? index - 1 : index;
    if (index > drag.from && without < drag.to) {
      return { transform: `translateY(${-drag.stride}px)` };
    }
    if (index < drag.from && without >= drag.to) {
      return { transform: `translateY(${drag.stride}px)` };
    }
    return undefined;
  };

  return (
    <ul ref={listRef} className="reorder-list" aria-label={ariaLabel}>
      {items.map((item, i) => {
        const key = itemKey(item);
        const dragging = drag?.key === key;
        return (
          <li
            key={key}
            className={`reorder-item ${itemClassName ?? ''}${dragging ? ' dragging' : ''}`}
            style={rowStyle(i)}
          >
            <button
              type="button"
              className="drag-handle"
              aria-label="Drag to reorder"
              title="Drag to reorder"
              disabled={items.length < 2}
              onPointerDown={(e) => startDrag(e, key, i)}
            >
              <svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor" aria-hidden="true">
                <circle cx="9" cy="5" r="1.7" />
                <circle cx="15" cy="5" r="1.7" />
                <circle cx="9" cy="12" r="1.7" />
                <circle cx="15" cy="12" r="1.7" />
                <circle cx="9" cy="19" r="1.7" />
                <circle cx="15" cy="19" r="1.7" />
              </svg>
            </button>
            {renderItem(item)}
            <span className="reorder-arrows">
              <button
                type="button"
                className="arrow-btn"
                aria-label="Move up"
                disabled={i === 0}
                onClick={() => move(i, i - 1)}
              >
                ▲
              </button>
              <button
                type="button"
                className="arrow-btn"
                aria-label="Move down"
                disabled={i === items.length - 1}
                onClick={() => move(i, i + 1)}
              >
                ▼
              </button>
            </span>
          </li>
        );
      })}
    </ul>
  );
}
