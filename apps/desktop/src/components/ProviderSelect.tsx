import { cloudOptions, groupedOptions, providerLabel, type ProviderOption } from "../lib/providers";
import type { ProviderCatalog, ProviderKind } from "../types";

interface ProviderSelectProps {
  kind: ProviderKind;
  value: string;
  onChange: (value: string) => void;
  /** Cloud entries that do not support this language are disabled. */
  sourceLanguage: string;
  /** `null` until `list_providers` has answered; the select then shows Auto
   *  plus the current value so it never renders empty. */
  catalog: ProviderCatalog | null;
  disabled?: boolean;
  id?: string;
  /** Only cloud providers, without groups (Auto-route preferences). */
  cloudOnly?: boolean;
}

/** One `<select>` for every provider choice in the app. Options come from
 *  the catalog so adding a provider to the registry needs no UI change. */
export function ProviderSelect({
  catalog,
  cloudOnly = false,
  disabled = false,
  id,
  kind,
  onChange,
  sourceLanguage,
  value,
}: ProviderSelectProps) {
  const groups = cloudOnly
    ? [{ label: "Cloud" as const, options: cloudOptions(catalog, kind, sourceLanguage) }]
    : groupedOptions(catalog, kind, sourceLanguage);
  const known = groups.some((group) => group.options.some((option) => option.value === value));
  const flat = groups.length === 1;
  return (
    <select id={id} value={value} disabled={disabled} onChange={(event) => onChange(event.target.value)}>
      {!known && <option value={value}>{providerLabel(catalog, kind, value)}</option>}
      {flat
        ? groups[0].options.map(renderOption)
        : groups.map((group) => (
            <optgroup key={group.label} label={group.label}>
              {group.options.map(renderOption)}
            </optgroup>
          ))}
    </select>
  );
}

function renderOption(option: ProviderOption) {
  return (
    <option key={option.value} value={option.value} disabled={option.disabled} title={option.reason}>
      {option.disabled && option.reason ? `${option.label} — ${option.reason}` : option.label}
    </option>
  );
}
