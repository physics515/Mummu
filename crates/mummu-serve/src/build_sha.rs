//! The rules by which a build names itself — which commit it came from, and
//! what has to change for that answer to be re-asked.
//!
//! # Why this is a module and not just `build.rs`
//!
//! It is compiled twice, on purpose. `build.rs` pulls it in with `#[path]` and
//! calls it at compile time; the crate pulls it in under `cfg(test)` so these
//! rules are *tested* rather than only asserted in a comment. A build script
//! is not a test target — nothing `cargo test` runs ever executes one — so a
//! rule that lives only inside `build.rs` is a rule nothing checks. That is
//! the wrong place for these particular rules: the value they produce is one
//! a human reads once, under pressure, to decide whether a deploy landed, and
//! the way it fails (a plausible-looking sha that is not the running code) is
//! invisible to everything except the person it misleads.

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
/// worse trade. So inside the image `git rev-parse` fails and there is
/// nothing left in the build to name the commit.
///
/// That is the failure this exists to prevent, not one it is recovering
/// from: the deployed v0.2.0 answers `/api/health` with no `build` field and
/// no `version` field at all — both are new here — which is why the question
/// "is the new release deployed?" could only be answered by going and looking
/// at what had been pushed. A field that says `unknown` would be no better.
/// So the sha is *passed in*, and what is passed in has to win, because there
/// is nothing inside the image for it to lose to.
///
/// # Why `-dirty` exists
///
/// A build made from uncommitted changes is not the commit it claims to be,
/// and "the sha matches but the behaviour does not" is the single most
/// confusing way for this field to mislead someone.
///
/// A preset is returned **verbatim**, marker and all: this function cannot
/// look at the tree the caller built from — in the Docker build there is no
/// repository inside the context to look at — so it cannot add the marker and
/// must not pretend the absence of one means clean. The dirty state therefore
/// has to be baked into what is passed in, which is what the deploy command in
/// `crates/mummu-serve/Dockerfile` does. That matters more there than it looks:
/// the compose build context is a COPY OF THE WORKING TREE, so a stamp naming
/// a bare commit really can describe an image the commit does not match.
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
/// # A stale sha with no marker is the worst value this field can carry
///
/// `unknown` is useless but honest, and `abc1234-dirty` says "do not trust
/// this to be that commit". A bare `abc1234` that is NOT `abc1234` is the one
/// answer that is confidently wrong, and it is the answer a missed watch
/// produces: the binary keeps the stamp from the last time the script ran, in
/// exactly the field someone is reading to decide whether their fix is live.
/// So the watch list is built for the awkward cases, not the common one.
///
/// # What actually has to be watched
///
/// Watching `HEAD` and the branch ref covers the ordinary commit: `HEAD` is a
/// *symbolic* ref that keeps pointing at the same branch, so what actually
/// changes underneath a commit is the branch's own file. But that file is
/// only a file while the ref is **loose**. `git gc` (and `git pack-refs`)
/// fold refs into `packed-refs` and delete the loose ones — and a path that
/// does not exist is not declared at all (`build.rs` checks first, because
/// cargo re-runs forever on a declared path that is missing). From then on
/// the branch ref is watched by nothing.
///
/// `packed-refs` catches the pack itself. `index` catches a commit, which
/// rewrites it whether refs are packed or loose. Between them they still miss
/// the case that closes this: a ref that MOVES without the index changing —
/// `git reset --soft`, `git branch -f`, `git update-ref`, a fast-forward that
/// touches no file this crate builds from. The ref moves, the loose file is
/// (re)created where nothing is looking, `packed-refs` is untouched because
/// creating a loose ref does not rewrite it, and the index is untouched
/// because no content changed. The stamp then names the previous commit.
///
/// The **reflogs** are what cover that, and they cover it by construction:
/// git appends to `logs/HEAD` and to `logs/<the ref>` on every ref update,
/// they are ordinary files that packing does not delete, and they exist in a
/// worktree the same way `HEAD` does. A repository with `core.logAllRefUpdates`
/// turned off falls back to the watches above, which is the behaviour this
/// had before and no worse.
#[must_use]
pub fn git_watches(head: Option<&str>) -> Vec<String> {
    let mut paths = vec!["HEAD".to_owned(), "ORIG_HEAD".to_owned()];
    if let Some(head) = preset(head) {
        paths.push(head.to_owned());
        // The ref's own reflog: appended to by every update, and still there
        // after a `git gc` has taken the loose ref away.
        paths.push(format!("logs/{head}"));
    }
    // And HEAD's, which moves for a checkout even when no branch does.
    paths.push("logs/HEAD".to_owned());
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

    /// The hole the pack watches left, and the reason it is worth closing: a
    /// ref can MOVE without the index moving (`reset --soft`, `branch -f`,
    /// `update-ref`, a fast-forward over files this crate does not build
    /// from). With the loose ref packed away — so `build.rs` never declares
    /// it, a missing path being a re-run-forever — and `packed-refs`
    /// untouched by the new loose ref, nothing else in the list changes and
    /// the binary keeps the PREVIOUS sha with no `-dirty` to hint at it.
    ///
    /// Reflogs are the watch that cannot be packed away: git appends to them
    /// on every ref update, and `git gc` does not delete them.
    #[test]
    fn the_watch_list_covers_a_ref_that_moves_without_the_index() {
        let w = git_watches(Some("refs/heads/main"));
        assert!(
            w.contains(&"logs/refs/heads/main".to_owned()),
            "a ref move that skips the index is only visible in its reflog: {w:?}"
        );
        assert!(
            w.contains(&"logs/HEAD".to_owned()),
            "and a checkout moves HEAD without moving any branch: {w:?}"
        );
    }

    /// A detached HEAD names a sha directly — there is no branch file to
    /// watch, and inventing one would declare a path that never exists. Its
    /// own reflog still moves, though, and that is what a `git checkout
    /// <sha>` writes to.
    #[test]
    fn a_detached_head_watches_no_branch_ref() {
        let w = git_watches(None);
        assert!(w.contains(&"HEAD".to_owned()));
        assert!(
            !w.iter().any(|p| p.starts_with("refs/")),
            "nothing to point at: {w:?}"
        );
        assert!(
            !w.iter().any(|p| p.starts_with("logs/refs/")),
            "and no branch reflog either: {w:?}"
        );
        assert!(w.contains(&"logs/HEAD".to_owned()), "{w:?}");
    }

    /// The deploy instructions are part of this module's contract, so they
    /// are checked like it.
    ///
    /// A comment is the only place the two rules that make this stamp
    /// trustworthy can be written down — [`stamp`] takes a preset verbatim,
    /// so the DIRTY STATE has to be in what the operator passes, and the
    /// build context is a copy of the working tree, so it genuinely can
    /// differ from the commit. A command that gets either wrong stamps a
    /// clean sha onto an image that is not that commit, which is the one
    /// outcome the marker exists to prevent.
    #[test]
    fn the_dockerfile_deploy_command_is_the_one_that_works_on_this_host() {
        let df = include_str!("../Dockerfile");
        for needle in [
            // The compose set that resolves on this host: a bare
            // `docker compose` there picks up the Windows one.
            "docker-compose.linux.yaml",
            "--env-file .env.linux",
            "build mummu",
            "up -d --no-deps mummu",
            // And the dirty state, which `stamp` will not add for you.
            "diff --quiet HEAD",
            "-dirty",
        ] {
            assert!(
                df.contains(needle),
                "the deploy comment has lost {needle:?} — it no longer describes a \
                 command that works, or one that stamps the truth"
            );
        }
    }

    /// MINOR 6: a comment that lies about live production is worse than no
    /// comment. v0.2.0 answers `/api/health` with NO version and NO build
    /// field — `build.rs` is new in v0.3.0 — so nothing here may claim it
    /// reports `"build":"unknown"`. Somebody reads that, curls the deployed
    /// container, sees neither field, and stops trusting the rest of the file.
    #[test]
    fn nothing_claims_the_deployed_release_answers_an_unknown_build() {
        for (name, text) in [
            ("Dockerfile", include_str!("../Dockerfile")),
            ("build_sha.rs", include_str!("build_sha.rs")),
        ] {
            let comments = text.lines().map(str::trim).filter(|l| {
                // Comments only: the claim being guarded against is a
                // COMMENT about production, and code that merely quotes this
                // rule (this test does) is not making it.
                l.starts_with('#') || l.starts_with("//")
            });
            for line in comments.filter(|l| l.contains("v0.2.0")) {
                // The two words may appear together only to DENY the claim,
                // which is what the correction itself has to say.
                assert!(
                    !line.contains("unknown") || line.contains("does NOT"),
                    "{name} claims v0.2.0 answers an unknown build: {line}"
                );
            }
            assert!(
                text.contains("no `build` field") || text.contains("no `version` field"),
                "{name} no longer says what v0.2.0 actually answers"
            );
        }
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
