use std::time::Instant;
fn main() {
    let list = std::env::args().nth(1).expect("list file");
    let paths: Vec<String> = std::fs::read_to_string(list).unwrap()
        .lines().map(|s| s.to_string()).collect();
    // warmup
    for p in &paths {
        let _ = sonara::core::audio::load(std::path::Path::new(p), 22050, true, 0.0, 0.0);
    }
    let mut best = f64::MAX;
    for _ in 0..3 {
        let t = Instant::now();
        for p in &paths {
            let _ = sonara::core::audio::load(std::path::Path::new(p), 22050, true, 0.0, 0.0).unwrap();
        }
        best = best.min(t.elapsed().as_secs_f64());
    }
    println!("best {best:.3}s for {} files", paths.len());
}
