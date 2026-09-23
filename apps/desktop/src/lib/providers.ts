import type {
  ProviderCatalog,
  ProviderKind,
  ProviderSpec,
  StartSessionRequest,
} from "../types";

/** Sentinel provider id: RuntimeRouter picks the provider at session start. */
export const AUTO_PROVIDER = "auto";
export const AUTO_LABEL = "Auto select";

export type ProviderGroupLabel = "Auto" | "Local" | "Cloud";

export interface ProviderOption {
  value: string;
  label: string;
  disabled: boolean;
  /** Why the option is disabled, for a title/tooltip. */
  reason?: string;
}

export interface ProviderOptionGroup {
  label: ProviderGroupLabel;
  options: ProviderOption[];
}

const languageNames: Record<string, string> = {
  en: "English",
  zh: "Chinese",
  ja: "Japanese",
  ko: "Korean",
};

/** Provider ids that never leave the machine. Used only while the catalog
 *  has not loaded yet; the catalog's flags win once it is available. */
const knownLocalIds = new Set(["qwen_local", "simulstreaming", "hymt_local", "mock", "none"]);

export function languageName(code: string): string {
  return languageNames[code] ?? code;
}

export function findProvider(
  catalog: ProviderCatalog | null | undefined,
  kind: ProviderKind,
  id: string,
): ProviderSpec | undefined {
  if (!catalog) return undefined;
  return catalog[kind].find((spec) => spec.id === id);
}

/** Human-readable label for a provider id, `auto` included. Falls back to
 *  the raw id when the catalog is missing or does not know the provider. */
export function providerLabel(
  catalog: ProviderCatalog | null | undefined,
  kind: ProviderKind,
  id: string,
): string {
  if (id === AUTO_PROVIDER) return AUTO_LABEL;
  return findProvider(catalog, kind, id)?.display_name ?? id;
}

/** Vendor name of a provider for privacy copy ("Audio is uploaded to
 *  Deepgram"). Falls back to the display name, then to the id. */
export function providerVendor(
  catalog: ProviderCatalog | null | undefined,
  kind: ProviderKind,
  id: string,
): string {
  const spec = findProvider(catalog, kind, id);
  return recipientName(spec?.vendor || spec?.display_name || id);
}

/** A vendor as the recipient in a sentence ("sent to …"). The registry's
 *  vendor for a self-hosted OpenAI-compatible endpoint is "Custom", which
 *  does not read as a recipient. */
export function recipientName(vendor: string): string {
  return vendor === "Custom" ? "your custom endpoint" : vendor;
}

/** Whether `spec` can be used for `lang` as source language. An empty
 *  language list means the provider is not language-restricted. */
export function supportsLanguage(spec: ProviderSpec, lang: string): boolean {
  return spec.languages.length === 0 || spec.languages.includes(lang);
}

export function providersForGroup(
  catalog: ProviderCatalog | null | undefined,
  groupId: string,
): { asr?: ProviderSpec; translation?: ProviderSpec } {
  if (!catalog) return {};
  return {
    asr: catalog.asr.find((spec) => spec.credential_group === groupId),
    translation: catalog.translation.find((spec) => spec.credential_group === groupId),
  };
}

/** The provider id a session would actually use for `kind`, or null when
 *  Auto may still pick a local provider (mode `auto`/`local`). */
export function effectiveProviderId(
  draft: Pick<
    StartSessionRequest,
    | "asr_provider"
    | "translation_provider"
    | "inference_mode"
    | "cloud_asr_preference"
    | "cloud_translation_preference"
  >,
  kind: ProviderKind,
): string | null {
  const selected = kind === "asr" ? draft.asr_provider : draft.translation_provider;
  if (selected !== AUTO_PROVIDER) return selected;
  if (draft.inference_mode !== "cloud") return null;
  const preference =
    kind === "asr" ? draft.cloud_asr_preference : draft.cloud_translation_preference;
  return preference || null;
}

function uploadRequired(
  catalog: ProviderCatalog | null | undefined,
  kind: ProviderKind,
  id: string | null,
  flag: "audio_upload_required" | "transcript_upload_required",
): boolean {
  if (!id) return false;
  const spec = findProvider(catalog, kind, id);
  if (spec) return spec[flag];
  // Catalog not loaded (or unknown id): anything that is not a known local
  // provider is treated as cloud so the privacy gate errs on the safe side.
  return !knownLocalIds.has(id);
}

/** True when the configured session would send processed audio off-device. */
export function needsAudioUpload(
  catalog: ProviderCatalog | null | undefined,
  draft: Parameters<typeof effectiveProviderId>[0],
): boolean {
  return uploadRequired(
    catalog,
    "asr",
    effectiveProviderId(draft, "asr"),
    "audio_upload_required",
  );
}

/** True when the configured session would send source transcript text
 *  off-device. Cloud ASR providers only ever receive audio. */
export function needsTranscriptUpload(
  catalog: ProviderCatalog | null | undefined,
  draft: Parameters<typeof effectiveProviderId>[0],
): boolean {
  return uploadRequired(
    catalog,
    "translation",
    effectiveProviderId(draft, "translation"),
    "transcript_upload_required",
  );
}

function languageRestriction(spec: ProviderSpec): string {
  return `${spec.languages.map(languageName).join(", ")} only`;
}

function toOption(spec: ProviderSpec, sourceLanguage: string): ProviderOption {
  const supported = supportsLanguage(spec, sourceLanguage);
  return {
    value: spec.id,
    label: spec.display_name,
    disabled: !supported,
    ...(supported ? {} : { reason: languageRestriction(spec) }),
  };
}

/** Options for a provider `<select>`: Auto, then on-device providers (local
 *  runtimes, then diagnostic mock/none entries), then cloud providers.
 *  Cloud entries that do not support `sourceLanguage` are disabled with a
 *  reason. Without a catalog only the Auto entry is returned. */
export function groupedOptions(
  catalog: ProviderCatalog | null | undefined,
  kind: ProviderKind,
  sourceLanguage: string,
): ProviderOptionGroup[] {
  const groups: ProviderOptionGroup[] = [
    { label: "Auto", options: [{ value: AUTO_PROVIDER, label: AUTO_LABEL, disabled: false }] },
  ];
  if (!catalog) return groups;
  const specs = catalog[kind].filter((spec) => spec.selectable);
  const local = specs.filter((spec) => spec.locality === "local");
  const diagnostics = specs.filter((spec) => spec.locality === "mock");
  const cloud = specs.filter((spec) => spec.locality === "cloud");
  const localOptions = [...local, ...diagnostics].map((spec) => toOption(spec, sourceLanguage));
  const cloudOptions = cloud.map((spec) => toOption(spec, sourceLanguage));
  if (localOptions.length > 0) groups.push({ label: "Local", options: localOptions });
  if (cloudOptions.length > 0) groups.push({ label: "Cloud", options: cloudOptions });
  return groups;
}

/** Cloud providers only, for the Auto-route preference selects. */
export function cloudOptions(
  catalog: ProviderCatalog | null | undefined,
  kind: ProviderKind,
  sourceLanguage: string,
): ProviderOption[] {
  return groupedOptions(catalog, kind, sourceLanguage).find((group) => group.label === "Cloud")
    ?.options ?? [];
}
