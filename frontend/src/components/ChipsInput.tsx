/** Small multi-value editor: toggleable suggestion chips + free-text add. */
import { useState } from 'react';

interface ChipsInputProps {
  value: string[];
  onChange: (next: string[]) => void;
  suggestions?: string[];
  placeholder?: string;
  id?: string;
  /**
   * Skip label normalization (lowercase + spaces→underscores) — for values
   * like host:port entries that must be preserved verbatim.
   */
  raw?: boolean;
  /**
   * Gate for free-text entries. A rejected draft stays in the box (with
   * `invalidHint` shown) so the user can correct it; suggestion chips are
   * trusted and bypass this.
   */
  validate?: (entry: string) => boolean;
  /** Short message shown while the current draft fails `validate`. */
  invalidHint?: string;
}

export default function ChipsInput({
  value,
  onChange,
  suggestions = [],
  placeholder = 'add…',
  id,
  raw = false,
  validate,
  invalidHint,
}: ChipsInputProps) {
  const [draft, setDraft] = useState('');
  const [invalid, setInvalid] = useState(false);
  const all = [...new Set([...suggestions, ...value])];

  const toggle = (item: string) => {
    onChange(value.includes(item) ? value.filter((v) => v !== item) : [...value, item]);
  };

  const addDraft = () => {
    const v = raw ? draft.trim() : draft.trim().toLowerCase().replace(/\s+/g, '_');
    if (!v) {
      setDraft('');
      setInvalid(false);
      return;
    }
    if (validate && !validate(v)) {
      setInvalid(true); // keep the draft so it can be corrected
      return;
    }
    if (!value.includes(v)) onChange([...value, v]);
    setDraft('');
    setInvalid(false);
  };

  return (
    <div className="chips" id={id}>
      {all.map((item) => (
        <button
          key={item}
          type="button"
          className={`chip ${value.includes(item) ? 'chip-on' : ''}`}
          onClick={() => toggle(item)}
          aria-pressed={value.includes(item)}
        >
          {item}
        </button>
      ))}
      <input
        className={`chip-input${invalid ? ' chip-input-invalid' : ''}`}
        value={draft}
        placeholder={placeholder}
        aria-invalid={invalid || undefined}
        onChange={(e) => {
          setDraft(e.target.value);
          if (invalid) setInvalid(false);
        }}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ',') {
            e.preventDefault();
            addDraft();
          }
        }}
        onBlur={addDraft}
        aria-label="Add value"
      />
      {invalid && invalidHint && <span className="form-error chip-error">{invalidHint}</span>}
    </div>
  );
}
