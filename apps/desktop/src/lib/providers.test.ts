import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { defaultSessionDefaults, type ProviderCatalog, type StartSessionRequest } from "../types";
import {
  cloudOptions,
  effectiveProviderId,
  groupedOptions,
  needsAudioUpload,
  needsTranscriptUpload,
  providerLabel,
  providersForGroup,
  providerVendor,
  recipientName,
  supportsLanguage,
} from "./providers";

// The committed catalog is the contract between the Python registry, the
// Rust shell and this UI; the tests run against the real file.
const catalog = JSON.parse(
  readFileSync(fileURLToPath(new URL("../../../../configs/providers.json", import.meta.url)), "utf8"),
) as ProviderCatalog;

const draft = (overrides: Partial<StartSessionRequest> = {}): StartSessionRequest => ({
  ...defaultSessionDefaults,
  ...overrides,
});

describe("provider catalog contract", () => {
  it("matches the shape the UI depends on", () => {
    expect(catalog.schema_version).toBe(1);
    for (const spec of [...catalog.asr, ...catalog.translation]) {
      expect(typeof spec.id).toBe("string");
      expect(["local", "cloud", "mock"]).toContain(spec.locality);
      expect(Array.isArray(spec.languages)).toBe(true);
      if (spec.locality === "cloud") {
        expect(spec.credential_group).not.toBeNull();
        expect(spec.vendor).not.toBe("");
      }
    }
    for (const group of catalog.credential_groups) {
      expect(group.fields.length).toBeGreaterThan(0);
      for (const setting of group.settings) {
        expect(["select", "text"]).toContain(setting.kind);
        if (setting.kind === "select") expect(setting.options.length).toBeGreaterThan(0);
      }
    }
  });

  it("links every cloud provider to a credential group in the catalog", () => {
    const groupIds = new Set(catalog.credential_groups.map((group) => group.id));
    for (const spec of [...catalog.asr, ...catalog.translation]) {
      if (spec.locality === "cloud") expect(groupIds.has(spec.credential_group ?? "")).toBe(true);
    }
  });
});

describe("providerLabel / providerVendor", () => {
  it("labels auto, catalog ids and unknown ids", () => {
    expect(providerLabel(catalog, "asr", "auto")).toBe("Auto select");
    expect(providerLabel(catalog, "asr", "qwen_local")).toBe("Qwen3-ASR (local)");
    expect(providerLabel(catalog, "translation", "deepl")).toBe("DeepL (cloud)");
    expect(providerLabel(catalog, "asr", "not_a_provider")).toBe("not_a_provider");
    expect(providerLabel(null, "asr", "qwen_cloud")).toBe("qwen_cloud");
  });

  it("uses the vendor name for privacy copy", () => {
    expect(providerVendor(catalog, "asr", "deepgram")).toBe("Deepgram");
    expect(providerVendor(catalog, "translation", "deepl")).toBe("DeepL");
    expect(providerVendor(catalog, "asr", "qwen_cloud")).toBe("Alibaba Cloud");
    expect(providerVendor(null, "asr", "deepgram")).toBe("deepgram");
    expect(recipientName("Custom")).toBe("your custom endpoint");
    expect(recipientName("DeepL")).toBe("DeepL");
  });
});

describe("providersForGroup / supportsLanguage", () => {
  it("resolves both roles of a shared credential group", () => {
    const dashscope = providersForGroup(catalog, "dashscope");
    expect(dashscope.asr?.id).toBe("qwen_cloud");
    expect(dashscope.translation?.id).toBe("qwen_cloud");
    const deepl = providersForGroup(catalog, "deepl");
    expect(deepl.asr).toBeUndefined();
    expect(deepl.translation?.id).toBe("deepl");
    expect(providersForGroup(catalog, "missing")).toEqual({ asr: undefined, translation: undefined });
    expect(providersForGroup(null, "deepl")).toEqual({});
  });

  it("treats an empty language list as unrestricted", () => {
    const assemblyai = catalog.asr.find((spec) => spec.id === "assemblyai")!;
    const deepl = catalog.translation.find((spec) => spec.id === "deepl")!;
    expect(supportsLanguage(assemblyai, "en")).toBe(true);
    expect(supportsLanguage(assemblyai, "zh")).toBe(false);
    expect(supportsLanguage(deepl, "ko")).toBe(true);
  });
});

describe("groupedOptions", () => {
  it("groups Auto, Local and Cloud in catalog order and skips unselectable entries", () => {
    const groups = groupedOptions(catalog, "asr", "en");
    expect(groups.map((group) => group.label)).toEqual(["Auto", "Local", "Cloud"]);
    expect(groups[0].options).toEqual([{ value: "auto", label: "Auto select", disabled: false }]);
    const local = groups[1].options.map((option) => option.value);
    expect(local).toEqual(["qwen_local", "simulstreaming", "mock"]);
    expect(local).not.toContain("none");
    const cloud = groups[2].options.map((option) => option.value);
    expect(cloud).toEqual(["qwen_cloud", "openai_realtime", "deepgram", "assemblyai", "gladia"]);
    expect(groups[2].options.every((option) => !option.disabled)).toBe(true);
  });

  it("disables cloud entries that do not support the source language, with a reason", () => {
    const cloud = groupedOptions(catalog, "asr", "zh").find((group) => group.label === "Cloud")!;
    const assemblyai = cloud.options.find((option) => option.value === "assemblyai")!;
    expect(assemblyai.disabled).toBe(true);
    expect(assemblyai.reason).toBe("English only");
    const deepgram = cloud.options.find((option) => option.value === "deepgram")!;
    expect(deepgram.disabled).toBe(false);
    expect(deepgram.reason).toBeUndefined();
  });

  it("keeps recognition-only and mock translators in the Local group", () => {
    const local = groupedOptions(catalog, "translation", "en").find((group) => group.label === "Local")!;
    expect(local.options.map((option) => option.value)).toEqual(["hymt_local", "none", "mock"]);
  });

  it("returns only Auto before the catalog is loaded", () => {
    expect(groupedOptions(null, "translation", "en")).toEqual([
      { label: "Auto", options: [{ value: "auto", label: "Auto select", disabled: false }] },
    ]);
  });

  it("exposes cloud-only options for the Auto route preferences", () => {
    const options = cloudOptions(catalog, "translation", "ja");
    expect(options.map((option) => option.value)).toContain("deepl");
    expect(options.every((option) => option.value !== "hymt_local" && option.value !== "auto")).toBe(true);
    expect(cloudOptions(null, "asr", "en")).toEqual([]);
  });
});

describe("upload requirements", () => {
  it("follows the catalog flags for explicit providers", () => {
    expect(needsAudioUpload(catalog, draft({ asr_provider: "deepgram" }))).toBe(true);
    expect(needsAudioUpload(catalog, draft({ asr_provider: "qwen_local" }))).toBe(false);
    expect(needsTranscriptUpload(catalog, draft({ translation_provider: "deepl" }))).toBe(true);
    expect(needsTranscriptUpload(catalog, draft({ translation_provider: "hymt_local" }))).toBe(false);
    // A cloud recognizer never needs transcript upload and vice versa.
    expect(needsTranscriptUpload(catalog, draft({ asr_provider: "deepgram" }))).toBe(false);
    expect(needsAudioUpload(catalog, draft({ translation_provider: "deepl" }))).toBe(false);
  });

  it("resolves Auto through the cloud preference only in Cloud mode", () => {
    const auto = draft({ inference_mode: "auto", cloud_asr_preference: "deepgram" });
    expect(effectiveProviderId(auto, "asr")).toBeNull();
    expect(needsAudioUpload(catalog, auto)).toBe(false);
    const cloud = draft({ inference_mode: "cloud", cloud_asr_preference: "deepgram", cloud_translation_preference: "deepl" });
    expect(effectiveProviderId(cloud, "asr")).toBe("deepgram");
    expect(effectiveProviderId(cloud, "translation")).toBe("deepl");
    expect(needsAudioUpload(catalog, cloud)).toBe(true);
    expect(needsTranscriptUpload(catalog, cloud)).toBe(true);
    const local = draft({ inference_mode: "local", cloud_asr_preference: "deepgram" });
    expect(needsAudioUpload(catalog, local)).toBe(false);
  });

  it("errs on the side of upload when the catalog is not loaded", () => {
    expect(needsAudioUpload(null, draft({ asr_provider: "qwen_cloud" }))).toBe(true);
    expect(needsAudioUpload(null, draft({ asr_provider: "some_new_cloud" }))).toBe(true);
    expect(needsAudioUpload(null, draft({ asr_provider: "qwen_local" }))).toBe(false);
    expect(needsAudioUpload(null, draft({ asr_provider: "auto", inference_mode: "cloud" }))).toBe(true);
    expect(needsTranscriptUpload(null, draft({ translation_provider: "hymt_local" }))).toBe(false);
    expect(needsTranscriptUpload(null, draft({ translation_provider: "auto", inference_mode: "cloud" }))).toBe(true);
  });
});
