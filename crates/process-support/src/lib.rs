//! Background child processes for the desktop shell.
//!
//! Every helper process (the inference sidecar, the local model services and
//! whatever they launch in turn) runs without a console window and never
//! outlives its owner: on Unix each child leads its own process group, on
//! Windows each child is placed in a kill-on-close Job Object, so stopping or
//! dropping a [`ProcessTree`] stops the whole tree, grandchildren included.
//!
//! On Windows every helper is also opted out of power throttling (see
//! [`disable_power_throttling_for_current_process`]).

use std::io;
use std::process::{ExitStatus, Output};
use std::time::Duration;
use tokio::process::{Child, Command};

/// How long [`ProcessTree::terminate`] waits on Windows for every process
/// of a terminated job to be gone.
#[cfg(windows)]
const JOB_EXIT_TIMEOUT: Duration = Duration::from_secs(5);

/// Prepare `command` for a background helper: no console window on Windows,
/// a fresh process group on Unix, and a kill when the handle is dropped.
pub fn configure_background(command: &mut Command) -> &mut Command {
    #[cfg(windows)]
    command.creation_flags(windows_sys::Win32::System::Threading::CREATE_NO_WINDOW);
    #[cfg(unix)]
    command.process_group(0);
    command.kill_on_drop(true)
}

/// Run a short-lived helper to completion and collect its output, like
/// [`Command::output`], with [`configure_background`] applied and power
/// throttling disabled on Windows. Unlike [`Command::output`], only the
/// streams the caller set to [`std::process::Stdio::piped`] are collected.
/// The helper is killed if the returned future is dropped (for example by a
/// timeout).
pub async fn background_output(command: &mut Command) -> io::Result<Output> {
    configure_background(command);
    let child = command.spawn()?;
    disable_power_throttling_for_child(&child);
    child.wait_with_output().await
}

/// Ask Windows not to throttle this process (EcoQoS): no lowered CPU
/// clocks or efficiency-core scheduling while it has no visible window or
/// runs on battery, and no coarser timers. Windows applies both to
/// windowless background processes, which slowed the local ASR and
/// translation services (and the GPU work they feed) by more than half. A
/// no-op elsewhere; callers treat a failure as a warning only.
pub fn disable_power_throttling_for_current_process() -> io::Result<()> {
    #[cfg(windows)]
    {
        // The pseudo handle needs no closing.
        let process = unsafe { windows_sys::Win32::System::Threading::GetCurrentProcess() };
        power::disable_throttling(process)
    }
    #[cfg(not(windows))]
    Ok(())
}

/// [`disable_power_throttling_for_current_process`] for a spawned helper.
/// Best effort: a failure is reported on stderr and the helper keeps running.
fn disable_power_throttling_for_child(child: &Child) {
    #[cfg(windows)]
    if let Some(process) = child.raw_handle() {
        if let Err(error) = power::disable_throttling(process.cast()) {
            eprintln!(
                "could not disable power throttling for process {}: {error}",
                child.id().unwrap_or_default()
            );
        }
    }
    #[cfg(not(windows))]
    let _ = child;
}

/// The power throttling requests, spelled out so their values can be
/// checked on every platform.
#[cfg(any(windows, test))]
mod throttling {
    /// One `PROCESS_POWER_THROTTLING_STATE`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(crate) struct PowerThrottlingState {
        pub(crate) version: u32,
        pub(crate) control_mask: u32,
        pub(crate) state_mask: u32,
    }

    /// `PROCESS_POWER_THROTTLING_CURRENT_VERSION`.
    pub(crate) const VERSION: u32 = 1;
    /// `PROCESS_POWER_THROTTLING_EXECUTION_SPEED`: EcoQoS (lowered clocks,
    /// efficiency cores).
    pub(crate) const EXECUTION_SPEED: u32 = 0x1;
    /// `PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION`: coarser timers.
    pub(crate) const IGNORE_TIMER_RESOLUTION: u32 = 0x4;

    /// The states to request, in order. A mechanism listed in the control
    /// mask but cleared in the state mask is turned off for the process.
    /// Windows 10 does not know the timer flag and rejects the whole
    /// request, so the execution-speed opt-out alone is the fallback.
    pub(crate) const OPT_OUTS: [PowerThrottlingState; 2] = [
        PowerThrottlingState {
            version: VERSION,
            control_mask: EXECUTION_SPEED | IGNORE_TIMER_RESOLUTION,
            state_mask: 0,
        },
        PowerThrottlingState {
            version: VERSION,
            control_mask: EXECUTION_SPEED,
            state_mask: 0,
        },
    ];
}

#[cfg(windows)]
mod power {
    use super::throttling::{PowerThrottlingState, OPT_OUTS};
    use std::ffi::c_void;
    use std::io;
    use windows_sys::Win32::Foundation::HANDLE;
    use windows_sys::Win32::System::Threading::{
        ProcessPowerThrottling, SetProcessInformation, PROCESS_POWER_THROTTLING_STATE,
    };

    impl From<PowerThrottlingState> for PROCESS_POWER_THROTTLING_STATE {
        fn from(state: PowerThrottlingState) -> Self {
            Self {
                Version: state.version,
                ControlMask: state.control_mask,
                StateMask: state.state_mask,
            }
        }
    }

    /// Apply the first opt-out Windows accepts; the error of the last one
    /// when it accepts none.
    pub(crate) fn disable_throttling(process: HANDLE) -> io::Result<()> {
        let mut result = Ok(());
        for state in OPT_OUTS {
            let state = PROCESS_POWER_THROTTLING_STATE::from(state);
            let applied = unsafe {
                SetProcessInformation(
                    process,
                    ProcessPowerThrottling,
                    (&state as *const PROCESS_POWER_THROTTLING_STATE).cast::<c_void>(),
                    std::mem::size_of::<PROCESS_POWER_THROTTLING_STATE>() as u32,
                )
            };
            if applied != 0 {
                return Ok(());
            }
            result = Err(io::Error::last_os_error());
        }
        result
    }
}

/// A spawned child together with every process it starts.
pub struct ProcessTree {
    child: Child,
    group: Group,
    finished: bool,
}

impl ProcessTree {
    /// Configure `command` with [`configure_background`], spawn it and take
    /// ownership of its process tree. On Windows the child is also opted out
    /// of power throttling (best effort). That does not reach the processes
    /// the child starts: a helper that launches the real workload (the
    /// `watch-process` wrapper) opts it out itself.
    pub fn spawn(command: &mut Command) -> io::Result<Self> {
        configure_background(command);
        let child = command.spawn()?;
        disable_power_throttling_for_child(&child);
        let group = Group::attach(&child);
        Ok(Self {
            child,
            group,
            finished: false,
        })
    }

    /// The OS process id of the direct child while it has not been reaped.
    pub fn id(&self) -> Option<u32> {
        self.child.id()
    }

    /// The direct child, for its standard streams.
    pub fn child_mut(&mut self) -> &mut Child {
        &mut self.child
    }

    pub fn try_wait(&mut self) -> io::Result<Option<ExitStatus>> {
        self.child.try_wait()
    }

    /// Wait for the direct child to exit on its own for at most `timeout`.
    pub async fn wait_timeout(&mut self, timeout: Duration) -> io::Result<Option<ExitStatus>> {
        match tokio::time::timeout(timeout, self.child.wait()).await {
            Ok(status) => status.map(Some),
            Err(_) => Ok(None),
        }
    }

    /// Stop the whole tree. On Unix the group first receives SIGTERM and gets
    /// `grace` to exit before SIGKILL; on Windows, which has no equivalent
    /// signal for windowless processes, the job is terminated at once (ask
    /// the process to exit through its own channel and [`Self::wait_timeout`]
    /// first when it supports a graceful shutdown). On Windows the call also
    /// waits (bounded) until the job is empty, because `TerminateJobObject`
    /// returns before the grandchildren have exited and released their files.
    pub async fn terminate(&mut self, grace: Duration) -> io::Result<ExitStatus> {
        #[cfg(unix)]
        {
            if self.child.try_wait()?.is_none() {
                self.group.signal(libc::SIGTERM);
                let _ = self.wait_timeout(grace).await?;
            }
            // Always sweep the group: the child may be gone while
            // grandchildren that ignored SIGTERM are still running.
            self.group.signal(libc::SIGKILL);
        }
        #[cfg(not(unix))]
        {
            let _ = grace;
            self.group.kill();
            if self.child.try_wait()?.is_none() {
                let _ = self.child.start_kill();
            }
        }
        let status = self.child.wait().await?;
        #[cfg(windows)]
        self.group.wait_until_empty(JOB_EXIT_TIMEOUT).await;
        self.finished = true;
        Ok(status)
    }

    /// Kill the whole tree immediately without waiting for it.
    pub fn kill(&mut self) {
        #[cfg(unix)]
        self.group.signal(libc::SIGKILL);
        #[cfg(not(unix))]
        self.group.kill();
        let _ = self.child.start_kill();
    }
}

impl Drop for ProcessTree {
    fn drop(&mut self) {
        if !self.finished {
            // `kill_on_drop` only reaches the direct child; the group or job
            // takes its descendants with it.
            self.kill();
        }
    }
}

#[cfg(unix)]
struct Group {
    process_group: Option<libc::pid_t>,
}

#[cfg(unix)]
impl Group {
    fn attach(child: &Child) -> Self {
        // `configure_background` made the child the leader of a new group
        // whose id is the child's pid.
        Self {
            process_group: child
                .id()
                .and_then(|id| libc::pid_t::try_from(id).ok())
                .filter(|id| *id > 0),
        }
    }

    fn signal(&self, signal: libc::c_int) {
        if let Some(group) = self.process_group {
            // ESRCH (the group is already gone) is the expected outcome of a
            // clean exit; nothing else can be done about other failures.
            unsafe {
                libc::killpg(group, signal);
            }
        }
    }
}

#[cfg(windows)]
struct Group {
    job: Option<job::Job>,
}

#[cfg(windows)]
impl Group {
    fn attach(child: &Child) -> Self {
        // A process that cannot be placed in a job (for example inside a job
        // that forbids nesting) is still killed directly.
        let job = child.raw_handle().and_then(|process| {
            let job = job::Job::kill_on_close().ok()?;
            job.assign(process.cast()).ok()?;
            Some(job)
        });
        Self { job }
    }

    fn kill(&self) {
        if let Some(job) = &self.job {
            let _ = job.terminate();
        }
    }

    /// Poll the job until no process is left in it, for at most `timeout`.
    async fn wait_until_empty(&self, timeout: Duration) {
        let Some(job) = &self.job else {
            return;
        };
        let deadline = tokio::time::Instant::now() + timeout;
        while job.active_processes().is_ok_and(|active| active > 0)
            && tokio::time::Instant::now() < deadline
        {
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    }
}

#[cfg(not(any(unix, windows)))]
struct Group;

#[cfg(not(any(unix, windows)))]
impl Group {
    fn attach(_child: &Child) -> Self {
        Self
    }

    fn kill(&self) {}
}

#[cfg(windows)]
mod job {
    use std::ffi::c_void;
    use std::io;
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, QueryInformationJobObject,
        SetInformationJobObject, TerminateJobObject, JobObjectBasicAccountingInformation,
        JobObjectExtendedLimitInformation, JOBOBJECT_BASIC_ACCOUNTING_INFORMATION,
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };

    /// An owned Job Object handle; closing it kills every process in the job.
    pub(crate) struct Job(HANDLE);

    // The handle is only used through thread-safe kernel calls.
    unsafe impl Send for Job {}
    unsafe impl Sync for Job {}

    impl Job {
        pub(crate) fn kill_on_close() -> io::Result<Self> {
            let handle = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
            if handle.is_null() {
                return Err(io::Error::last_os_error());
            }
            let job = Self(handle);
            let mut limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
            limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            let configured = unsafe {
                SetInformationJobObject(
                    job.0,
                    JobObjectExtendedLimitInformation,
                    (&limits as *const JOBOBJECT_EXTENDED_LIMIT_INFORMATION).cast::<c_void>(),
                    std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                )
            };
            if configured == 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(job)
        }

        pub(crate) fn assign(&self, process: HANDLE) -> io::Result<()> {
            if unsafe { AssignProcessToJobObject(self.0, process) } == 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        }

        pub(crate) fn terminate(&self) -> io::Result<()> {
            if unsafe { TerminateJobObject(self.0, 1) } == 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        }

        pub(crate) fn active_processes(&self) -> io::Result<u32> {
            let mut accounting = JOBOBJECT_BASIC_ACCOUNTING_INFORMATION::default();
            let queried = unsafe {
                QueryInformationJobObject(
                    self.0,
                    JobObjectBasicAccountingInformation,
                    (&mut accounting as *mut JOBOBJECT_BASIC_ACCOUNTING_INFORMATION)
                        .cast::<c_void>(),
                    std::mem::size_of::<JOBOBJECT_BASIC_ACCOUNTING_INFORMATION>() as u32,
                    std::ptr::null_mut(),
                )
            };
            if queried == 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(accounting.ActiveProcesses)
        }
    }

    impl Drop for Job {
        fn drop(&mut self) {
            unsafe {
                CloseHandle(self.0);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    fn process_alive(pid: u32) -> bool {
        let output = std::process::Command::new("ps")
            .args(["-o", "stat=", "-p", &pid.to_string()])
            .output()
            .expect("ps is available");
        let state = String::from_utf8_lossy(&output.stdout);
        let state = state.trim();
        // A reparented zombie is dead; only its reaper has not caught up.
        !state.is_empty() && !state.starts_with('Z')
    }

    #[cfg(unix)]
    async fn wait_until_gone(pid: u32) -> bool {
        for _ in 0..40 {
            if !process_alive(pid) {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        false
    }

    /// Spawn a shell that starts a background `sleep` (the grandchild) and
    /// reports its pid.
    #[cfg(unix)]
    async fn spawn_with_grandchild(script: &str) -> (ProcessTree, u32) {
        use tokio::io::AsyncBufReadExt;

        let mut command = Command::new("/bin/sh");
        command
            .args(["-c", script])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped());
        let mut tree = ProcessTree::spawn(&mut command).unwrap();
        let stdout = tree.child_mut().stdout.take().unwrap();
        let mut line = String::new();
        tokio::io::BufReader::new(stdout)
            .read_line(&mut line)
            .await
            .unwrap();
        let grandchild = line.trim().parse::<u32>().unwrap();
        assert!(process_alive(grandchild));
        (tree, grandchild)
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn terminate_stops_the_grandchild_gracefully() {
        let (mut tree, grandchild) = spawn_with_grandchild("sleep 30 & echo $!; wait").await;
        let started = std::time::Instant::now();
        tree.terminate(Duration::from_secs(5)).await.unwrap();
        // SIGTERM is enough: no need to wait for the grace period.
        assert!(started.elapsed() < Duration::from_secs(4));
        assert!(wait_until_gone(grandchild).await, "grandchild survived");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn terminate_escalates_when_the_tree_ignores_sigterm() {
        let (mut tree, grandchild) =
            spawn_with_grandchild("trap '' TERM; sleep 30 & echo $!; wait").await;
        tree.terminate(Duration::from_millis(300)).await.unwrap();
        assert!(wait_until_gone(grandchild).await, "grandchild survived");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn dropping_the_tree_kills_the_grandchild() {
        let (tree, grandchild) = spawn_with_grandchild("sleep 30 & echo $!; wait").await;
        drop(tree);
        assert!(wait_until_gone(grandchild).await, "grandchild survived");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn wait_timeout_reports_a_clean_exit() {
        let mut command = Command::new("/bin/sh");
        command.args(["-c", "exit 3"]);
        let mut tree = ProcessTree::spawn(&mut command).unwrap();
        let status = tree
            .wait_timeout(Duration::from_secs(5))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(status.code(), Some(3));
    }

    #[test]
    fn power_throttling_is_turned_off_with_a_windows_10_fallback() {
        use throttling::{PowerThrottlingState, OPT_OUTS};

        assert_eq!(
            OPT_OUTS,
            [
                PowerThrottlingState {
                    version: 1,
                    control_mask: 0x5,
                    state_mask: 0,
                },
                PowerThrottlingState {
                    version: 1,
                    control_mask: 0x1,
                    state_mask: 0,
                },
            ]
        );
    }

    #[cfg(windows)]
    #[test]
    fn power_throttling_values_match_the_windows_headers() {
        use windows_sys::Win32::System::Threading::{
            PROCESS_POWER_THROTTLING_CURRENT_VERSION, PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
            PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION, PROCESS_POWER_THROTTLING_STATE,
        };

        assert_eq!(
            throttling::VERSION,
            PROCESS_POWER_THROTTLING_CURRENT_VERSION
        );
        assert_eq!(
            throttling::EXECUTION_SPEED,
            PROCESS_POWER_THROTTLING_EXECUTION_SPEED
        );
        assert_eq!(
            throttling::IGNORE_TIMER_RESOLUTION,
            PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION
        );
        let state = PROCESS_POWER_THROTTLING_STATE::from(throttling::OPT_OUTS[0]);
        assert_eq!(
            (state.Version, state.ControlMask, state.StateMask),
            (1, 0x5, 0)
        );
    }

    #[test]
    fn the_current_process_can_be_opted_out() {
        // A no-op outside Windows; on Windows the call must succeed.
        disable_power_throttling_for_current_process().unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn background_output_collects_the_helper_output() {
        let mut command = Command::new("/bin/sh");
        command
            .args(["-c", "echo out; echo err >&2; exit 2"])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        let output = background_output(&mut command).await.unwrap();
        assert_eq!(output.status.code(), Some(2));
        assert_eq!(output.stdout, b"out\n");
        assert_eq!(output.stderr, b"err\n");
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn background_output_collects_the_helper_output() {
        let mut command = Command::new("cmd");
        command
            .args(["/c", "echo out& exit 2"])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        let output = background_output(&mut command).await.unwrap();
        assert_eq!(output.status.code(), Some(2));
        assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), "out");
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn terminate_empties_the_job() {
        let mut command = Command::new("cmd");
        command
            .args(["/c", "ping", "-n", "30", "127.0.0.1"])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null());
        let mut tree = ProcessTree::spawn(&mut command).unwrap();
        let job = tree.group.job.as_ref().expect("child was placed in a job");
        // cmd starts ping as a grandchild inside the same job.
        let mut active = 0;
        for _ in 0..100 {
            active = job.active_processes().unwrap();
            if active >= 2 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert!(active >= 2, "expected cmd and ping in the job, saw {active}");
        tree.terminate(Duration::from_millis(300)).await.unwrap();
        // terminate() only returns once the grandchild is gone as well.
        let job = tree.group.job.as_ref().unwrap();
        assert_eq!(job.active_processes().unwrap(), 0);
    }
}
