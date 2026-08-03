/**
 * Press-and-hold button with two modes:
 *  - hold-to-confirm (siren): `onConfirm` fires once after the button is held
 *    for `holdMs`; releasing early cancels. A fill animates progress via a
 *    CSS custom property.
 *  - momentary (push-to-talk): pass `onPressStart`/`onPressEnd` — they fire
 *    immediately on press and release, with no confirm delay.
 * Works with mouse and touch (pointer events + pointer capture, so releasing
 * outside the button still ends the press) and keyboard (hold Space).
 */
import { useCallback, useEffect, useRef, useState } from 'react';

interface HoldButtonProps {
  holdMs?: number;
  /** Hold-to-confirm mode: fires once after holdMs of holding. */
  onConfirm?: () => void;
  /** Momentary mode: fires as soon as the button is pressed. */
  onPressStart?: () => void;
  /** Momentary mode: fires when the press is released or cancelled. */
  onPressEnd?: () => void;
  disabled?: boolean;
  className?: string;
  /** Hint line under the label (confirm mode default: "hold to trigger"). */
  hint?: string;
  children: React.ReactNode;
}

export default function HoldButton({
  holdMs = 1200,
  onConfirm,
  onPressStart,
  onPressEnd,
  disabled,
  className,
  hint,
  children,
}: HoldButtonProps) {
  const momentary = onPressStart !== undefined || onPressEnd !== undefined;
  const [progress, setProgress] = useState(0);
  const rafRef = useRef(0);
  const startRef = useRef(0);
  const firedRef = useRef(false);
  const pressedRef = useRef(false);
  // Latest release callback, so unmount cleanup never calls a stale closure.
  const endRef = useRef(onPressEnd);
  endRef.current = onPressEnd;

  const cancelProgress = useCallback(() => {
    cancelAnimationFrame(rafRef.current);
    startRef.current = 0;
    setProgress(0);
  }, []);

  const tick = useCallback(() => {
    if (!startRef.current) return;
    const p = Math.min(1, (performance.now() - startRef.current) / holdMs);
    setProgress(p);
    if (p >= 1) {
      if (!firedRef.current) {
        firedRef.current = true;
        onConfirm?.();
      }
      cancelProgress();
      return;
    }
    rafRef.current = requestAnimationFrame(tick);
  }, [holdMs, onConfirm, cancelProgress]);

  const press = useCallback(() => {
    if (disabled || pressedRef.current) return;
    pressedRef.current = true;
    if (momentary) {
      onPressStart?.();
      return;
    }
    firedRef.current = false;
    startRef.current = performance.now();
    rafRef.current = requestAnimationFrame(tick);
  }, [disabled, momentary, onPressStart, tick]);

  const release = useCallback(() => {
    if (!pressedRef.current) return;
    pressedRef.current = false;
    if (momentary) {
      onPressEnd?.();
      return;
    }
    cancelProgress();
  }, [momentary, onPressEnd, cancelProgress]);

  useEffect(
    () => () => {
      cancelAnimationFrame(rafRef.current);
      // Unmounting mid-press must still release (stops PTT mic capture).
      if (pressedRef.current) {
        pressedRef.current = false;
        endRef.current?.();
      }
    },
    [],
  );

  return (
    <button
      type="button"
      className={`hold-btn ${className ?? ''}`}
      style={{ ['--hold' as never]: progress }}
      disabled={disabled}
      onPointerDown={(e) => {
        try {
          e.currentTarget.setPointerCapture(e.pointerId);
        } catch {
          /* capture unsupported — leave/cancel handlers still end the press */
        }
        press();
      }}
      onPointerUp={release}
      onPointerLeave={release}
      onPointerCancel={release}
      onKeyDown={(e) => {
        if ((e.key === ' ' || e.key === 'Spacebar') && !e.repeat) {
          e.preventDefault();
          press();
        }
      }}
      onKeyUp={(e) => {
        if (e.key === ' ' || e.key === 'Spacebar') {
          e.preventDefault();
          release();
        }
      }}
      onBlur={release}
      onContextMenu={(e) => e.preventDefault()}
      aria-label={momentary ? undefined : 'hold to confirm'}
    >
      <span className="hold-btn-label">{children}</span>
      <span className="hold-btn-hint">
        {momentary ? hint : progress > 0 ? 'keep holding…' : hint ?? 'hold to trigger'}
      </span>
    </button>
  );
}
