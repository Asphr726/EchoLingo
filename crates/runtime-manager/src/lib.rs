//! Managed, pinned local-model installation for the desktop product.

use hf_hub::progress::{DownloadEvent, ProgressEvent, ProgressHandler};
use hf_hub::HFClient;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use thiserror::Error;

pub const MODEL_CATALOG_VERSION: u16 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelRole {
    Asr,
    Translation,
    Alignment,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum DownloadKind {
    Snapshot { allow_patterns: Vec<String> },
    File { filename: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSpec {
    pub id: String,
    pub display_name: String,
    pub role: ModelRole,
    pub repository: String,
    pub revision: String,
    pub expected_bytes: u64,
    pub required_file: String,
    pub required_file_sha256: String,
    download: DownloadKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelInstallState {
    NotDownloaded,
    Installing,
    Ready,
    Corrupt,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ModelStatus {
    pub id: String,
    pub display_name: String,
    pub role: ModelRole,
    pub size_bytes: u64,
    pub state: ModelInstallState,
    pub path: PathBuf,
    pub revision: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ModelProgress {
    pub model_id: String,
    pub bytes_completed: u64,
    pub total_bytes: u64,
    pub bytes_per_second: Option<f64>,
    pub phase: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct InstallManifest {
    schema_version: u16,
    model_id: String,
    repository: String,
    revision: String,
    required_file: String,
    required_file_sha256: String,
}

#[derive(Debug, Error)]
pub enum ModelManagerError {
    #[error("unknown local model: {0}")]
    UnknownModel(String),
    #[error("model installation is already running: {0}")]
    AlreadyInstalling(String),
    #[error("not enough disk space: need {required_bytes} bytes, have {available_bytes} bytes")]
    DiskSpace {
        required_bytes: u64,
        available_bytes: u64,
    },
    #[error("model download failed: {0}")]
    Download(String),
    #[error("model integrity check failed for {0}")]
    Integrity(String),
    #[error("model filesystem operation failed: {0}")]
    Io(#[from] io::Error),
    #[error("model metadata is invalid: {0}")]
    Metadata(#[from] serde_json::Error),
}

type ProgressCallback = Arc<dyn Fn(ModelProgress) + Send + Sync>;

pub struct ModelManager {
    root: PathBuf,
    cache: PathBuf,
    active: Mutex<HashSet<String>>,
}

impl ModelManager {
    pub fn new(root: PathBuf) -> Self {
        let cache = root.join(".downloads");
        Self {
            root,
            cache,
            active: Mutex::new(HashSet::new()),
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn list(&self) -> Vec<ModelStatus> {
        catalog()
            .into_iter()
            .map(|spec| self.status_for(&spec))
            .collect()
    }

    pub fn status(&self, model_id: &str) -> Result<ModelStatus, ModelManagerError> {
        let spec = model_spec(model_id)?;
        Ok(self.status_for(&spec))
    }

    fn status_for(&self, spec: &ModelSpec) -> ModelStatus {
        let path = self.root.join(&spec.id);
        let active = self
            .active
            .lock()
            .is_ok_and(|active| active.contains(&spec.id));
        let state = if active {
            ModelInstallState::Installing
        } else if !path.exists() {
            ModelInstallState::NotDownloaded
        } else if installed_manifest_matches(&path, spec) && required_file_matches_size(&path, spec)
        {
            ModelInstallState::Ready
        } else {
            ModelInstallState::Corrupt
        };
        ModelStatus {
            id: spec.id.clone(),
            display_name: spec.display_name.clone(),
            role: spec.role,
            size_bytes: spec.expected_bytes,
            state,
            path,
            revision: spec.revision.clone(),
        }
    }

    pub async fn install(
        &self,
        model_id: &str,
        callback: ProgressCallback,
    ) -> Result<ModelStatus, ModelManagerError> {
        let spec = model_spec(model_id)?;
        {
            let mut active = self
                .active
                .lock()
                .map_err(|_| io::Error::other("model manager lock poisoned"))?;
            if !active.insert(spec.id.clone()) {
                return Err(ModelManagerError::AlreadyInstalling(spec.id));
            }
        }
        let result = self.install_inner(&spec, callback).await;
        if let Ok(mut active) = self.active.lock() {
            active.remove(&spec.id);
        }
        result.map(|()| self.status_for(&spec))
    }

    async fn install_inner(
        &self,
        spec: &ModelSpec,
        callback: ProgressCallback,
    ) -> Result<(), ModelManagerError> {
        std::fs::create_dir_all(&self.root)?;
        std::fs::create_dir_all(&self.cache)?;
        let required = spec.expected_bytes.saturating_add(spec.expected_bytes / 10);
        let available = fs2::available_space(&self.root)?;
        if available < required {
            return Err(ModelManagerError::DiskSpace {
                required_bytes: required,
                available_bytes: available,
            });
        }

        let staging = self.root.join(format!(".{}.installing", spec.id));
        remove_scoped_directory(&self.root, &staging)?;
        std::fs::create_dir_all(&staging)?;
        callback(ModelProgress {
            model_id: spec.id.clone(),
            bytes_completed: 0,
            total_bytes: spec.expected_bytes,
            bytes_per_second: None,
            phase: "starting".into(),
        });

        let client = HFClient::builder()
            .cache_dir(&self.cache)
            .user_agent(format!("echolingo/{}", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|error| ModelManagerError::Download(error.to_string()))?;
        let (owner, name) = spec
            .repository
            .split_once('/')
            .ok_or_else(|| ModelManagerError::Download("invalid repository name".into()))?;
        let repository = client.model(owner, name);
        let progress = ModelProgressHandler {
            model_id: spec.id.clone(),
            callback: callback.clone(),
            files: Mutex::new(HashMap::new()),
        };
        let download_result = match &spec.download {
            DownloadKind::Snapshot { allow_patterns } => repository
                .snapshot_download()
                .revision(spec.revision.clone())
                .allow_patterns(allow_patterns.clone())
                .local_dir(staging.clone())
                .max_workers(4)
                .progress(progress)
                .send()
                .await
                .map(|_| ()),
            DownloadKind::File { filename } => repository
                .download_file()
                .filename(filename.clone())
                .revision(spec.revision.clone())
                .local_dir(staging.clone())
                .progress(progress)
                .send()
                .await
                .map(|_| ()),
        };
        if let Err(error) = download_result {
            remove_scoped_directory(&self.root, &staging)?;
            return Err(ModelManagerError::Download(error.to_string()));
        }

        callback(ModelProgress {
            model_id: spec.id.clone(),
            bytes_completed: spec.expected_bytes,
            total_bytes: spec.expected_bytes,
            bytes_per_second: None,
            phase: "verifying".into(),
        });
        verify_required_file(&staging, spec)?;
        write_manifest(&staging, spec)?;
        activate_directory(&self.root, &staging, &self.root.join(&spec.id))?;
        callback(ModelProgress {
            model_id: spec.id.clone(),
            bytes_completed: spec.expected_bytes,
            total_bytes: spec.expected_bytes,
            bytes_per_second: None,
            phase: "ready".into(),
        });
        Ok(())
    }

    pub fn verify(&self, model_id: &str) -> Result<ModelStatus, ModelManagerError> {
        let spec = model_spec(model_id)?;
        verify_required_file(&self.root.join(&spec.id), &spec)?;
        Ok(self.status_for(&spec))
    }

    pub fn delete(&self, model_id: &str) -> Result<ModelStatus, ModelManagerError> {
        let spec = model_spec(model_id)?;
        if self
            .active
            .lock()
            .is_ok_and(|active| active.contains(&spec.id))
        {
            return Err(ModelManagerError::AlreadyInstalling(spec.id));
        }
        remove_scoped_directory(&self.root, &self.root.join(&spec.id))?;
        Ok(self.status_for(&spec))
    }
}

struct ModelProgressHandler {
    model_id: String,
    callback: ProgressCallback,
    files: Mutex<HashMap<String, (u64, u64)>>,
}

impl ProgressHandler for ModelProgressHandler {
    fn on_progress(&self, event: &ProgressEvent) {
        let progress = match event {
            ProgressEvent::Download(DownloadEvent::Start { total_bytes, .. }) => ModelProgress {
                model_id: self.model_id.clone(),
                bytes_completed: 0,
                total_bytes: *total_bytes,
                bytes_per_second: None,
                phase: "downloading".into(),
            },
            ProgressEvent::Download(DownloadEvent::Progress { files }) => {
                let mut cumulative = match self.files.lock() {
                    Ok(value) => value,
                    Err(_) => return,
                };
                for file in files {
                    cumulative.insert(
                        file.filename.clone(),
                        (file.bytes_completed, file.total_bytes),
                    );
                }
                ModelProgress {
                    model_id: self.model_id.clone(),
                    bytes_completed: cumulative.values().map(|value| value.0).sum(),
                    total_bytes: cumulative.values().map(|value| value.1).sum(),
                    bytes_per_second: None,
                    phase: "downloading".into(),
                }
            }
            ProgressEvent::Download(DownloadEvent::AggregateProgress {
                bytes_completed,
                total_bytes,
                bytes_per_sec,
            }) => ModelProgress {
                model_id: self.model_id.clone(),
                bytes_completed: *bytes_completed,
                total_bytes: *total_bytes,
                bytes_per_second: *bytes_per_sec,
                phase: "downloading".into(),
            },
            ProgressEvent::Download(DownloadEvent::Complete) => ModelProgress {
                model_id: self.model_id.clone(),
                bytes_completed: 0,
                total_bytes: 0,
                bytes_per_second: None,
                phase: "downloaded".into(),
            },
            _ => return,
        };
        (self.callback)(progress);
    }
}

fn model_spec(model_id: &str) -> Result<ModelSpec, ModelManagerError> {
    catalog()
        .into_iter()
        .find(|spec| spec.id == model_id)
        .ok_or_else(|| ModelManagerError::UnknownModel(model_id.into()))
}

pub fn catalog() -> Vec<ModelSpec> {
    let qwen_files = vec![
        "*.json".into(),
        "*.txt".into(),
        "*.md".into(),
        "model.safetensors".into(),
    ];
    vec![
        ModelSpec {
            id: "qwen3-asr-0.6b".into(),
            display_name: "Qwen3-ASR 0.6B".into(),
            role: ModelRole::Asr,
            repository: "Qwen/Qwen3-ASR-0.6B".into(),
            revision: "5eb144179a02acc5e5ba31e748d22b0cf3e303b0".into(),
            expected_bytes: 1_876_091_704,
            required_file: "model.safetensors".into(),
            required_file_sha256:
                "79d6cbd4c98c7bbffe9db2edac07f56cd6637d0d5944b27f6c2b8353840323ea".into(),
            download: DownloadKind::Snapshot {
                allow_patterns: qwen_files.clone(),
            },
        },
        ModelSpec {
            id: "qwen3-forced-aligner-0.6b".into(),
            display_name: "Qwen3 ForcedAligner 0.6B".into(),
            role: ModelRole::Alignment,
            repository: "Qwen/Qwen3-ForcedAligner-0.6B".into(),
            revision: "c7cbfc2048c462b0d63a45797104fc9db3ad62b7".into(),
            expected_bytes: 1_835_544_544,
            required_file: "model.safetensors".into(),
            required_file_sha256:
                "47831d0e82f96b20e9034dba01a075ee06436654719f6a68289e49f1b65ce0e7".into(),
            download: DownloadKind::Snapshot {
                allow_patterns: qwen_files,
            },
        },
        ModelSpec {
            id: "hymt2-1.8b".into(),
            display_name: "Hy-MT2 1.8B Q4_K_M".into(),
            role: ModelRole::Translation,
            repository: "tencent/Hy-MT2-1.8B-GGUF".into(),
            revision: "1cd5208700acedef4ef93019b6cfc148b8522d45".into(),
            expected_bytes: 1_133_080_448,
            required_file: "Hy-MT2-1.8B-Q4_K_M.gguf".into(),
            required_file_sha256:
                "dc5f44fcf1fa496ee7ad725982c0c8c553a4de00259b53af84c4b89fb0c06699".into(),
            download: DownloadKind::File {
                filename: "Hy-MT2-1.8B-Q4_K_M.gguf".into(),
            },
        },
    ]
}

fn manifest_path(directory: &Path) -> PathBuf {
    directory.join(".echolingo-model.json")
}

fn installed_manifest_matches(directory: &Path, spec: &ModelSpec) -> bool {
    std::fs::read(manifest_path(directory))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<InstallManifest>(&bytes).ok())
        .is_some_and(|manifest| {
            manifest.schema_version == MODEL_CATALOG_VERSION
                && manifest.model_id == spec.id
                && manifest.repository == spec.repository
                && manifest.revision == spec.revision
                && manifest.required_file == spec.required_file
                && manifest.required_file_sha256 == spec.required_file_sha256
        })
}

fn required_file_matches_size(directory: &Path, spec: &ModelSpec) -> bool {
    std::fs::metadata(directory.join(&spec.required_file))
        .is_ok_and(|metadata| metadata.len() == spec.expected_bytes)
}

fn verify_required_file(directory: &Path, spec: &ModelSpec) -> Result<(), ModelManagerError> {
    let path = directory.join(&spec.required_file);
    if !required_file_matches_size(directory, spec) {
        return Err(ModelManagerError::Integrity(spec.id.clone()));
    }
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    if format!("{:x}", digest.finalize()) != spec.required_file_sha256 {
        return Err(ModelManagerError::Integrity(spec.id.clone()));
    }
    Ok(())
}

fn write_manifest(directory: &Path, spec: &ModelSpec) -> Result<(), ModelManagerError> {
    let manifest = InstallManifest {
        schema_version: MODEL_CATALOG_VERSION,
        model_id: spec.id.clone(),
        repository: spec.repository.clone(),
        revision: spec.revision.clone(),
        required_file: spec.required_file.clone(),
        required_file_sha256: spec.required_file_sha256.clone(),
    };
    let temporary = directory.join(".echolingo-model.json.tmp");
    std::fs::write(&temporary, serde_json::to_vec_pretty(&manifest)?)?;
    std::fs::rename(temporary, manifest_path(directory))?;
    Ok(())
}

fn activate_directory(root: &Path, staging: &Path, destination: &Path) -> io::Result<()> {
    let previous = root.join(format!(
        ".{}.previous",
        destination
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("model")
    ));
    remove_scoped_directory(root, &previous)?;
    if destination.exists() {
        std::fs::rename(destination, &previous)?;
    }
    if let Err(error) = std::fs::rename(staging, destination) {
        if previous.exists() {
            let _ = std::fs::rename(&previous, destination);
        }
        return Err(error);
    }
    remove_scoped_directory(root, &previous)
}

fn remove_scoped_directory(root: &Path, target: &Path) -> io::Result<()> {
    if !target.exists() {
        return Ok(());
    }
    let root = root.canonicalize()?;
    let parent = target
        .parent()
        .ok_or_else(|| io::Error::other("model path has no parent"))?
        .canonicalize()?;
    if parent != root {
        return Err(io::Error::other(
            "refusing to remove path outside model root",
        ));
    }
    std::fs::remove_dir_all(target)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_is_pinned_and_has_integrity_metadata() {
        for spec in catalog() {
            assert_eq!(spec.revision.len(), 40);
            assert_eq!(spec.required_file_sha256.len(), 64);
            assert!(spec.expected_bytes > 1_000_000_000);
        }
    }

    #[test]
    fn unknown_model_cannot_escape_managed_root() {
        let directory = tempfile::tempdir().unwrap();
        let manager = ModelManager::new(directory.path().join("models"));
        assert!(matches!(
            manager.delete("../../outside"),
            Err(ModelManagerError::UnknownModel(_))
        ));
    }

    #[test]
    fn incomplete_model_is_reported_as_corrupt_and_can_be_deleted() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("models");
        let manager = ModelManager::new(root.clone());
        std::fs::create_dir_all(root.join("qwen3-asr-0.6b")).unwrap();
        assert_eq!(
            manager.status("qwen3-asr-0.6b").unwrap().state,
            ModelInstallState::Corrupt
        );
        assert_eq!(
            manager.delete("qwen3-asr-0.6b").unwrap().state,
            ModelInstallState::NotDownloaded
        );
    }
}
