//! Capture the git short sha at compile time, so a running server can answer
//! "is the new release deployed?" by itself.
//!
//! The question was asked and could not be answered: `/api/health` reported
//! the device and the adapters and nothing about the build, the workspace
//! version sat at 0.1.0 through three tags, and the only way to tell which
//! commit a container was running was to go and look at what had been pushed.
//! A version alone would not have settled it either — two builds of 0.3.0 from
//! either side of a fix are the same string. The sha is what identifies a
//! build.
//!
//! # This build script must never fail
//!
//! It runs in contexts where there is no git and no `.git` at all: a Docker
//! build that `COPY`s the sources in, a `cargo package` tarball, a vendored
//! source tree. In every one of those the right answer is the string
//! `unknown`, printed into the binary, and an exit status of zero. So nothing
//! here returns an error, unwraps, or looks at git's exit code as anything but
//! a hint — a build that fell over because it could not name itself would be
//! a strictly worse outcome than a build that does not know its own sha.

use std::process::Command;

fn main() {
    // Without this, cargo re-runs the script on every change to the crate,
    // which is harmless but noisy; with the HEAD watches below it re-runs
    // exactly when the sha could have changed.
    println!("cargo:rerun-if-changed=build.rs");
    watch_head();
    println!("cargo:rustc-env=MUMMU_BUILD_SHA={}", short_sha());
}

/// The short sha, or `unknown`. Also marks a dirty tree, because a build made
/// from uncommitted changes is not the commit it claims to be — and "the sha
/// matches but the behaviour does not" is the single most confusing way for
/// this field to mislead someone.
fn short_sha() -> String {
    let Some(sha) = git(&["rev-parse", "--short=7", "HEAD"]) else {
        return "unknown".to_owned();
    };
    // `--quiet` makes this exit non-zero when there is anything staged or
    // modified; an empty diff exits zero. A repository too broken to answer
    // is treated as clean rather than smeared with a misleading marker.
    let dirty = Command::new("git")
        .args(["diff", "--quiet", "HEAD"])
        .status()
        .is_ok_and(|s| !s.success());
    if dirty { format!("{sha}-dirty") } else { sha }
}

/// Run git, returning trimmed stdout only when it actually succeeded.
///
/// Three separate ways this legitimately comes back `None`: git is not
/// installed, there is no repository here, or the repository has no commits
/// yet. None of them is an error worth failing a build over.
fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8(out.stdout).ok()?.trim().to_owned();
    (!text.is_empty()).then_some(text)
}

/// Re-run when HEAD moves.
///
/// `git rev-parse --git-path HEAD` rather than a hardcoded `../../.git/HEAD`,
/// because this crate is routinely built from a **worktree**, where `.git` is
/// a file pointing elsewhere and the real HEAD lives under the main
/// repository's `worktrees/` directory. Asking git where its own files are is
/// the only way to watch the right ones.
fn watch_head() {
    for path in ["HEAD", "ORIG_HEAD"] {
        if let Some(p) = git(&["rev-parse", "--git-path", path])
            && std::path::Path::new(&p).exists()
        {
            println!("cargo:rerun-if-changed={p}");
        }
    }
    // A detached HEAD names a sha directly; an attached one names a ref whose
    // file is what actually changes on a commit.
    if let Some(r) = git(&["symbolic-ref", "--quiet", "HEAD"])
        && let Some(p) = git(&["rev-parse", "--git-path", &r])
        && std::path::Path::new(&p).exists()
    {
        println!("cargo:rerun-if-changed={p}");
    }
}
