//! The rules by which a build names itself — which commit it came from, and
//! what has to change for that answer to be re-asked.
//!
//! # Why this is a module and not just `build.rs`
//!
//! It is compiled twice, on purpose. `build.rs` pulls it in with `#[path]` and
//! calls it at compile time; the crate pulls it in under `cfg(test)` so these
//! rules are *tested* rather than only asserted in a comment. A build script
//! is not a test target — nothing `cargo test` runs ever executes one — so a
//! rule that lives only inside `build.rs` is a rule nothing checks, which is
//! exactly how the production image came to stamp itself `unknown` for a
//! whole release without anybody noticing until a deploy had to be confirmed.

/// What a build that cannot name itself is called.
///
/// A string, not an absence: every consumer of `/api/health` wants a value in
/// that field, and "the build is called unknown" is at least a fact. What it
/// must never be is the *only* thing this can produce — see [`stamp`].
pub const UNKNOWN: &str = "unknown";

/// The explicit stamp, if one was set, with surrounding whitespace removed.
///
/// Separate from [`stamp`] because `build.rs` asks it first: a preset means
/// the git calls can be skipped entirely, and `git diff HEAD` is a stat of
/// every file in the working tree — wasted in the one environment (the Docker
/// build) this exists for, where there is no working tree to stat.
///
/// Whitespace-only is treated as unset. `--build-arg MUMMU_BUILD_SHA=` and a
/// shell that expanded an empty variable both arrive here as `Some("")`, and
/// neither is someone naming a build.
#[must_use]
pub fn preset(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|s| !s.is_empty())
}

/// Resolve the stamp `/api/health` and the status object report, from the
/// three things that can name a build.
///
/// # Why a preset outranks git
///
/// `crates/mummu-serve/Dockerfile` does `COPY . .` from a context whose
/// `.dockerignore` excludes `.git/` — deliberately, because the repository is
/// history the image has no use for and shipping it to fix this would be a
/// worse trade. So inside the image `git rev-parse` fails, and on origin/main
/// that meant the released container answered `"build":"unknown"`: the owner
/// deployed, curled `/api/health` to confirm the deploy had landed, and got
/// the one answer that identifies nothing. The sha has to be *passed in*, and
/// what is passed in has to win, because there is nothing there to lose to.
///
/// # Why `-dirty` exists
///
/// A build made from uncommitted changes is not the commit it claims to be,
/// and "the sha matches but the behaviour does not" is the single most
/// confusing way for this field to mislead someone. The marker is only
/// meaningful for a build made *from* a tree, so a preset never carries it:
/// there is no tree there to call clean or dirty.
#[must_use]
pub fn stamp(preset_value: Option<&str>, git: Option<&str>, dirty: bool) -> String {
    if let Some(explicit) = preset(preset_value) {
        return explicit.to_owned();
    }
    match preset(git) {
        Some(sha) if dirty => format!("{sha}-dirty"),
        Some(sha) => sha.to_owned(),
        // No repository and nobody said. `unknown` and an exit status of
        // zero: a build that fell over because it could not name itself
        // would be strictly worse than one that does not know its own sha.
        None => UNKNOWN.to_owned(),
    }
}

/// The git files whose change can change the stamp, as names to hand to
/// `git rev-parse --git-path`.
///
/// `head` is the ref `HEAD` points at (`refs/heads/main`), or `None` on a
/// detached HEAD — which names a sha directly and so needs no ref watched.
///
/// # The two cases that used to leave a stale sha
///
/// Watching `HEAD` and the branch ref covers the ordinary commit: `HEAD` is a
/// *symbolic* ref that keeps pointing at the same branch, so what actually
/// changes underneath a commit is the branch's own file. But that file is
/// only a file while the ref is **loose**. `git gc` (and `git pack-refs`)
/// fold refs into `packed-refs` and delete the loose ones, and from then on
/// nothing the build script declared changes when a commit is made: the
/// branch ref reappears as a file cargo was never told about, `HEAD` is
/// untouched, and the next build keeps the sha it had — stale, and *without*
/// the `-dirty` marker that exists to make exactly this mismatch obvious.
///
/// `index` is the watch that closes it, and it is the robust one rather than
/// the clever one: every commit rewrites the index, packed refs or not, as
/// does every `git add` and every checkout.
#[must_use]
pub fn git_watches(head: Option<&str>) -> Vec<String> {
    let mut paths = vec!["HEAD".to_owned(), "ORIG_HEAD".to_owned()];
    if let Some(head) = preset(head) {
        paths.push(head.to_owned());
    }
    paths.push("packed-refs".to_owned());
    paths.push("index".to_owned());
    paths
}

/// The working-tree paths whose change makes the tree dirty *in a way that
/// changes this binary*, relative to the repository root.
///
/// # Why the working tree has to be declared at all
///
/// Nothing inside `.git` moves when a source file is edited — `git diff HEAD`
/// reads the working tree directly. So a clean build followed by an edit
/// recompiled the crate (cargo watches the crate's own sources for *that*)
/// while the build script did **not** re-run, and the resulting binary
/// carried the previous, clean sha with no `-dirty` marker. The marker cannot
/// appear unless something tells cargo to ask again.
///
/// # Why the sources and not the whole repository
///
/// Two reasons, and the first is the honest one: a README or ROADMAP edit
/// dirties the tree without changing a byte of the binary, and re-stamping
/// the build for it would make `-dirty` mean "someone was typing" instead of
/// "this binary is not its commit". The second is practical — `target/` sits
/// at the repository root and handing cargo a directory to walk recursively
/// had better not be that one.
#[must_use]
pub const fn tree_watches() -> [&'static str; 3] {
    ["Cargo.toml", "Cargo.lock", "crates"]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// MAJOR: the Docker image has sources but no repository, so git answers
    /// nothing and the passed-in sha is the only thing that can name the
    /// build. It has to win, and it has to win *whatever* git says, because
    /// the whole point is that git is not there to be asked.
    #[test]
    fn an_explicit_stamp_outranks_git() {
        assert_eq!(stamp(Some("e528e2a"), None, false), "e528e2a");
        assert_eq!(
            stamp(Some("e528e2a"), Some("8a44bd1"), true),
            "e528e2a",
            "a preset is not a fallback; it is the answer"
        );
    }

    /// An unset build-arg reaches the build script as an empty string, not as
    /// an absence. Taking that literally would stamp the binary with nothing
    /// at all — worse than `unknown`, because it reads as a missing field
    /// rather than as a build nobody named.
    #[test]
    fn an_empty_or_blank_stamp_is_not_a_stamp() {
        assert_eq!(preset(Some("")), None);
        assert_eq!(preset(Some("   \n")), None);
        assert_eq!(preset(None), None);
        assert_eq!(stamp(Some(""), Some("8a44bd1"), false), "8a44bd1");
        assert_eq!(stamp(Some("  "), None, false), UNKNOWN);
        // And a stamp that merely has a newline on it (a `$(git rev-parse)`
        // that kept its trailing newline) is still a stamp.
        assert_eq!(stamp(Some(" e528e2a\n"), None, false), "e528e2a");
    }

    /// The marker that stops "the sha matches but the behaviour does not".
    #[test]
    fn a_dirty_tree_is_marked_and_a_clean_one_is_not() {
        assert_eq!(stamp(None, Some("8a44bd1"), true), "8a44bd1-dirty");
        assert_eq!(stamp(None, Some("8a44bd1"), false), "8a44bd1");
    }

    /// The build must not fail because it cannot name itself, so the last
    /// resort is a string rather than an error — but it is the LAST resort.
    #[test]
    fn unknown_is_only_reached_when_nothing_at_all_answers() {
        assert_eq!(stamp(None, None, false), UNKNOWN);
        assert_eq!(stamp(None, None, true), UNKNOWN, "no sha, nothing to dirty");
    }

    /// MINOR 7(a): the branch ref is a file only until `git gc` packs it. The
    /// two watches that survive a pack are what keep a commit from producing
    /// a binary stamped with the previous commit.
    #[test]
    fn the_watch_list_survives_a_packed_ref() {
        let w = git_watches(Some("refs/heads/main"));
        assert!(w.contains(&"refs/heads/main".to_owned()), "{w:?}");
        assert!(
            w.contains(&"packed-refs".to_owned()),
            "a gc that packs the ref must still re-stamp: {w:?}"
        );
        assert!(
            w.contains(&"index".to_owned()),
            "every commit rewrites the index, packed or loose: {w:?}"
        );
        assert!(w.contains(&"HEAD".to_owned()), "{w:?}");
    }

    /// A detached HEAD names a sha directly — there is no branch file to
    /// watch, and inventing one would declare a path that never exists.
    #[test]
    fn a_detached_head_watches_no_branch_ref() {
        let w = git_watches(None);
        assert!(w.contains(&"HEAD".to_owned()));
        assert!(
            !w.iter().any(|p| p.starts_with("refs/")),
            "nothing to point at: {w:?}"
        );
    }

    /// MINOR 7(b): editing a source file touches nothing under `.git`, so the
    /// sources themselves are the only thing that can make the build script
    /// ask `git diff` again.
    #[test]
    fn the_watch_list_covers_the_sources_that_make_a_tree_dirty() {
        let w = tree_watches();
        assert!(w.contains(&"crates"), "{w:?}");
        assert!(w.contains(&"Cargo.lock"), "a dependency bump is a change");
        assert!(
            !w.contains(&"target") && !w.contains(&"."),
            "cargo walks a declared directory; never hand it the build output"
        );
    }
}
