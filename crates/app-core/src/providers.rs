//! Provider catalog embedded from `configs/providers.json`.
//!
//! `src/echolingo/backends/registry.py` is the single source of truth for
//! provider ids, credential fields, privacy flags and settings; it exports the
//! checked-in JSON that this module parses at first use. The Rust shell never
//! hard-codes a provider id: privacy gating, credential storage and settings
//! validation all consult this catalog.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt;
use std::sync::OnceLock;

/// The raw catalog exactly as exported by the Python registry.
pub const PROVIDER_CATALOG_JSON: &str = include_str!("../../../configs/providers.json");

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    Asr,
    Translation,
}

impl ProviderKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ProviderKind::Asr => "asr",
            ProviderKind::Translation => "translation",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "asr" => Some(ProviderKind::Asr),
            "translation" => Some(ProviderKind::Translation),
            _ => None,
        }
    }
}

impl fmt::Display for ProviderKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Locality {
    Local,
    Cloud,
    Mock,
}

impl Locality {
    pub fn as_str(self) -> &'static str {
        match self {
            Locality::Local => "local",
            Locality::Cloud => "cloud",
            Locality::Mock => "mock",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderSpec {
    pub id: String,
    pub kind: ProviderKind,
    pub locality: Locality,
    pub display_name: String,
    #[serde(default)]
    pub vendor: String,
    #[serde(default)]
    pub credential_group: Option<String>,
    #[serde(default)]
    pub audio_upload_required: bool,
    #[serde(default)]
    pub transcript_upload_required: bool,
    #[serde(default)]
    pub languages: Vec<String>,
    #[serde(default)]
    pub local_service_id: Option<String>,
    #[serde(default)]
    pub auto_route_eligible: bool,
    #[serde(default = "default_true")]
    pub streaming_partials: bool,
    #[serde(default = "default_true")]
    pub selectable: bool,
    #[serde(default)]
    pub description: String,
}

impl ProviderSpec {
    pub fn is_cloud(&self) -> bool {
        self.locality == Locality::Cloud
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CredentialField {
    pub key: String,
    pub env_var: String,
    pub label: String,
    #[serde(default = "default_true")]
    pub secret: bool,
    #[serde(default = "default_true")]
    pub required: bool,
    #[serde(default)]
    pub min_len: usize,
    pub keychain_account: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderSetting {
    pub key: String,
    pub label: String,
    /// `select` or `text`.
    #[serde(default = "default_setting_kind")]
    pub kind: String,
    /// `(value, label)` pairs; only meaningful for `select`.
    #[serde(default)]
    pub options: Vec<(String, String)>,
    #[serde(default)]
    pub default: String,
    #[serde(default, deserialize_with = "deserialize_nullable_string")]
    pub env_var: String,
    #[serde(default)]
    pub placeholder: String,
}

impl ProviderSetting {
    pub fn is_select(&self) -> bool {
        self.kind == "select"
    }

    /// The environment variable this setting reaches the sidecar through, if
    /// any. Empty strings in the catalog mean "UI only".
    pub fn env_var(&self) -> Option<&str> {
        if self.env_var.is_empty() {
            None
        } else {
            Some(self.env_var.as_str())
        }
    }

    pub fn accepts(&self, value: &str) -> bool {
        if !self.is_select() {
            return true;
        }
        self.options.iter().any(|(option, _)| option == value)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CredentialGroup {
    pub id: String,
    pub display_name: String,
    #[serde(default)]
    pub vendor: String,
    #[serde(default)]
    pub docs_url: String,
    #[serde(default)]
    pub free_tier_note: String,
    #[serde(default)]
    pub fields: Vec<CredentialField>,
    #[serde(default)]
    pub settings: Vec<ProviderSetting>,
}

impl CredentialGroup {
    pub fn field(&self, key: &str) -> Option<&CredentialField> {
        self.fields.iter().find(|field| field.key == key)
    }

    pub fn setting(&self, key: &str) -> Option<&ProviderSetting> {
        self.settings.iter().find(|setting| setting.key == key)
    }
}

/// A chat provider the AI assistant can use for notes and titles
/// (docs/adr/0006). `group_id` names the credential group whose key it uses.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssistantPreset {
    pub group_id: String,
    pub display_name: String,
    /// The model used when the preference leaves the model empty. May be
    /// empty for endpoints that name their own model (custom OpenAI).
    #[serde(default)]
    pub default_model: String,
    #[serde(default)]
    pub models: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderCatalog {
    pub schema_version: u16,
    #[serde(default)]
    pub asr: Vec<ProviderSpec>,
    #[serde(default)]
    pub translation: Vec<ProviderSpec>,
    #[serde(default)]
    pub credential_groups: Vec<CredentialGroup>,
    /// Chat presets for the AI assistant; absent in catalogs exported before
    /// docs/adr/0006.
    #[serde(default)]
    pub assistant: Vec<AssistantPreset>,
}

impl ProviderCatalog {
    pub fn parse(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    pub fn specs(&self, kind: ProviderKind) -> &[ProviderSpec] {
        match kind {
            ProviderKind::Asr => &self.asr,
            ProviderKind::Translation => &self.translation,
        }
    }

    pub fn find(&self, kind: ProviderKind, id: &str) -> Option<&ProviderSpec> {
        self.specs(kind).iter().find(|spec| spec.id == id)
    }

    pub fn group(&self, id: &str) -> Option<&CredentialGroup> {
        self.credential_groups.iter().find(|group| group.id == id)
    }

    /// The credential group a provider needs, if it needs one.
    pub fn group_for(&self, kind: ProviderKind, id: &str) -> Option<&CredentialGroup> {
        self.find(kind, id)
            .and_then(|spec| spec.credential_group.as_deref())
            .and_then(|group| self.group(group))
    }

    /// The assistant preset for a credential group, if the group can chat.
    pub fn assistant_preset(&self, group_id: &str) -> Option<&AssistantPreset> {
        self.assistant.iter().find(|preset| preset.group_id == group_id)
    }

    pub fn cloud_ids(&self, kind: ProviderKind) -> impl Iterator<Item = &str> {
        self.specs(kind)
            .iter()
            .filter(|spec| spec.is_cloud())
            .map(|spec| spec.id.as_str())
    }
}

/// The embedded catalog, parsed once. The checked-in JSON is validated by a
/// unit test, so a parse failure here is a build defect rather than runtime
/// input; the panic message never includes user data.
pub fn catalog() -> &'static ProviderCatalog {
    static CATALOG: OnceLock<ProviderCatalog> = OnceLock::new();
    CATALOG.get_or_init(|| {
        ProviderCatalog::parse(PROVIDER_CATALOG_JSON)
            .expect("configs/providers.json must match the app-core catalog schema")
    })
}

/// The embedded catalog as raw JSON for UI consumers, exactly as exported by
/// the Python registry (sorted keys, no Rust-side reshaping).
pub fn catalog_value() -> Value {
    serde_json::from_str(PROVIDER_CATALOG_JSON)
        .expect("configs/providers.json must be valid JSON")
}

/// SHA-256 of the embedded catalog, computed the way the Python registry
/// computes `catalog_digest()` (the JSON text with exactly one trailing
/// newline), so it can be compared with the sidecar's `hello_accepted`
/// `providers_digest` to detect a stale `configs/providers.json`.
pub fn catalog_digest() -> &'static str {
    static DIGEST: OnceLock<String> = OnceLock::new();
    DIGEST.get_or_init(|| {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(PROVIDER_CATALOG_JSON.trim_end().as_bytes());
        hasher.update(b"\n");
        hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    })
}

fn default_true() -> bool {
    true
}

fn default_setting_kind() -> String {
    "text".into()
}

fn deserialize_nullable_string<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(Option::<String>::deserialize(deserializer)?.unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_catalog_parses_and_has_the_default_route() {
        let catalog = catalog();
        assert_eq!(catalog.schema_version, 1);
        let qwen_local = catalog.find(ProviderKind::Asr, "qwen_local").unwrap();
        assert_eq!(qwen_local.locality, Locality::Local);
        assert!(!qwen_local.audio_upload_required);
        assert!(qwen_local.auto_route_eligible);
        let hymt = catalog
            .find(ProviderKind::Translation, "hymt_local")
            .unwrap();
        assert!(!hymt.transcript_upload_required);
        assert!(catalog.find(ProviderKind::Asr, "hymt_local").is_none());
        assert!(catalog.find(ProviderKind::Asr, "auto").is_none());
    }

    #[test]
    fn cloud_specs_carry_privacy_flags_and_credential_groups() {
        let catalog = catalog();
        let asr = catalog.find(ProviderKind::Asr, "qwen_cloud").unwrap();
        assert!(asr.is_cloud() && asr.audio_upload_required);
        assert_eq!(asr.credential_group.as_deref(), Some("dashscope"));
        let translation = catalog
            .find(ProviderKind::Translation, "qwen_cloud")
            .unwrap();
        assert!(translation.transcript_upload_required && !translation.audio_upload_required);
        for kind in [ProviderKind::Asr, ProviderKind::Translation] {
            for spec in catalog.specs(kind) {
                if spec.is_cloud() {
                    let group = spec
                        .credential_group
                        .as_deref()
                        .unwrap_or_else(|| panic!("{} {} has no credential group", kind, spec.id));
                    assert!(catalog.group(group).is_some(), "missing group {group}");
                    assert!(
                        spec.audio_upload_required || spec.transcript_upload_required,
                        "{} {} uploads nothing?",
                        kind,
                        spec.id
                    );
                } else {
                    assert!(!spec.audio_upload_required && !spec.transcript_upload_required);
                }
            }
        }
    }

    #[test]
    fn credential_groups_expose_keychain_accounts_and_settings() {
        let dashscope = catalog().group("dashscope").unwrap();
        let api_key = dashscope.field("api_key").unwrap();
        assert_eq!(api_key.keychain_account, "dashscope-api-key");
        assert_eq!(api_key.env_var, "DASHSCOPE_API_KEY");
        assert!(api_key.secret && api_key.required && api_key.min_len == 8);
        let workspace = dashscope.field("workspace_id").unwrap();
        assert_eq!(workspace.keychain_account, "dashscope-workspace-id");
        assert!(!workspace.required);
        let region = dashscope.setting("region").unwrap();
        assert!(region.is_select());
        assert_eq!(region.default, "singapore");
        assert_eq!(region.env_var(), Some("ECHOLINGO_QWEN_REGION"));
        assert!(region.accepts("beijing") && !region.accepts("mars"));
        let custom = catalog().group("custom_openai").unwrap();
        assert!(!custom.field("api_key").unwrap().required);
        assert_eq!(custom.setting("base_url").unwrap().default, "");
        for group in &catalog().credential_groups {
            for field in &group.fields {
                assert!(!field.keychain_account.is_empty());
                assert!(!field.env_var.is_empty());
            }
        }
    }

    #[test]
    fn catalog_digest_is_a_stable_sha256_hex() {
        let digest = catalog_digest();
        assert_eq!(digest.len(), 64);
        assert!(digest.chars().all(|c| c.is_ascii_hexdigit()));
        assert_eq!(digest, catalog_digest());
        // Trailing whitespace in the checked-in file never changes the digest,
        // which mirrors the registry's `catalog_json()` (one trailing newline).
        if let Ok(expected) = std::env::var("ECHOLINGO_EXPECTED_CATALOG_DIGEST") {
            assert_eq!(digest, expected);
        }
    }

    #[test]
    fn assistant_presets_are_optional_and_reference_credential_groups() {
        // Catalogs exported before docs/adr/0006 have no `assistant` array.
        let legacy = ProviderCatalog::parse(
            r#"{"schema_version": 1, "asr": [], "translation": [], "credential_groups": []}"#,
        )
        .unwrap();
        assert!(legacy.assistant.is_empty());
        assert!(legacy.assistant_preset("dashscope").is_none());

        let current = ProviderCatalog::parse(
            r#"{"schema_version": 1, "credential_groups": [], "assistant": [
                {"group_id": "dashscope", "display_name": "Qwen (Alibaba Model Studio)",
                 "default_model": "qwen-plus", "models": ["qwen-plus", "qwen-max"]},
                {"group_id": "custom_openai", "display_name": "Custom endpoint"}
            ]}"#,
        )
        .unwrap();
        let qwen = current.assistant_preset("dashscope").unwrap();
        assert_eq!(qwen.default_model, "qwen-plus");
        assert_eq!(qwen.models, vec!["qwen-plus", "qwen-max"]);
        let custom = current.assistant_preset("custom_openai").unwrap();
        assert!(custom.default_model.is_empty() && custom.models.is_empty());

        // Whatever the embedded catalog lists must use an existing key card.
        for preset in &catalog().assistant {
            assert!(
                catalog().group(&preset.group_id).is_some(),
                "assistant preset {} has no credential group",
                preset.group_id
            );
            assert!(!preset.display_name.is_empty());
        }
    }

    #[test]
    fn raw_catalog_value_matches_the_parsed_shape() {
        let value = catalog_value();
        assert_eq!(value["schema_version"], 1);
        assert_eq!(
            value["asr"].as_array().unwrap().len(),
            catalog().asr.len()
        );
        assert_eq!(
            value["credential_groups"].as_array().unwrap().len(),
            catalog().credential_groups.len()
        );
    }
}
