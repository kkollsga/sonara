# MP3 decode scan harness (Part A gate, 2026-07-19)
- skip_probe.rs: symphonia-direct per-file CSV scan (path, ok_packets, skipped,
  fatal, secs). Files with skipped>0 are exactly those pre-0.2.5 sonara rejected.
  Build as a bin crate with workspace symphonia (0.5, all-codecs/all-formats).
- decode_bench.rs: wall-clock A/B of sonara::core::audio::load over a path list
  (bin crate with sonara path dep; swap the dep path for pre/post builds).
Gate evidence 2026-07-19: 2000-file random sample → 51 would-fail (2.6%,
extrapolates ≈915/35,898 ~ sonagram's 947); post-fix analyze_batch: 51/51
recovered; healthy-25 decode wall time 2.537s pre vs 2.538s post (noise).
