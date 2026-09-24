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
//! source tree. In every one of those the right answer is a sha handed in
//! through `MUMMU_BUILD_SHA` — or, failing that, the string `unknown` — and an
//! exit status of zero. So nothing here returns an error, unwraps, or looks at
//! git's exit code as anything but a hint: a build that fell over because it
//! could not name itself would be a strictly worse outcome than a build that
//! does not know its own sha.
//!
//! The rules themselves live in `src/build_sha.rs`, which the crate also
//! compiles under `cfg(test)`. Nothing `cargo test` runs executes a build
//! script, so anything decided here and only here is decided untested.

#![warn(clippy::pedantic, clippy::nursery, clippy::all)]

use std::path::Path;
use std::process::Command;

#[path = "src/build_sha.rs"]
mod build_sha;

fn main() {
    // Without these, cargo re-runs the script on every change to the crate,
    // which is harmless but noisy; with the watches below it re-runs exactly
    // when the sha could have changed.
    println!("cargo:rerun-if-changed=build.rs");
    // The escape hatch is itself an input: a deploy that passes a new sha must
    // re-stamp the binary even though not one source byte moved.
    println!("cargo:rerun-if-env-changed=MUMMU_BUILD_SHA");
    watch();
    println!("cargo:rustc-env=MUMMU_BUILD_SHA={}", resolve());
}

/// Gather [`build_sha::stamp`]'s three inputs from the environment and git.
fn resolve() -> String {
    let env = std::env::var("MUMMU_BUILD_SHA").ok();
    // Asked first so a preset costs no git at all. That is not a micro
    // optimization: the dirty check below stats every file in the working
    // tree, and the environment this exists for — the Docker build — has no
    // working tree and no git binary to run against it.
    if build_sha::preset(env.as_deref()).is_some() {
        return build_sha::stamp(env.as_deref(), None, false);
    }
    let sha = git(&["rev-parse", "--short=7", "HEAD"]);
    // `--quiet` makes this exit non-zero when there is anything staged or
    // modified; an empty diff exits zero. A repository too broken to answer
    // is treated as clean rather than smeared with a misleading marker.
    let dirty = sha.is_some()
        && Command::new("git")
            .args(["diff", "--quiet", "HEAD"])
            .status()
            .is_ok_and(|s| !s.success());
    build_sha::stamp(None, sha.as_deref(), dirty)
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

/// Declare everything whose change can change the stamp.
///
/// Paths are resolved with `git rev-parse --git-path` rather than hardcoded
/// as `../../.git/...`, because this crate is routinely built from a
/// **worktree**, where `.git` is a file pointing elsewhere and HEAD, the
/// index and `logs/HEAD` live under the main repository's `worktrees/`
/// directory while `packed-refs`, the refs themselves and their reflogs stay
/// in the common one. Asking git where its own files are is the only way to
/// watch the right ones — and it is why the reflog watches work at all here.
///
/// Every path is checked for existence first. Cargo treats a declared path
/// that is not there as a reason to re-run forever, and `ORIG_HEAD`, a packed
/// branch ref, and the reflogs of a repository with `core.logAllRefUpdates`
/// off genuinely may not exist.
fn watch() {
    let head = git(&["symbolic-ref", "--quiet", "HEAD"]);
    for name in build_sha::git_watches(head.as_deref()) {
        if let Some(p) = git(&["rev-parse", "--git-path", &name])
            && Path::new(&p).exists()
        {
            println!("cargo:rerun-if-changed={p}");
        }
    }
    // And the working tree, which is where dirtiness lives and which nothing
    // under `.git` reflects — see `build_sha::tree_watches`.
    if let Some(top) = git(&["rev-parse", "--show-toplevel"]) {
        for rel in build_sha::tree_watches() {
            let p = Path::new(&top).join(rel);
            if p.exists() {
                println!("cargo:rerun-if-changed={}", p.display());
            }
        }
    }
}
