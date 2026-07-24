use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::HashSet;
use std::f32::consts::PI;
use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::Array1;
use sonara::aggression;
use sonara::analyze::{self, AnalysisConfig};
use sonara::types::Float;

struct TrackingAllocator;

static ALLOCATION_COUNT: AtomicUsize = AtomicUsize::new(0);
static ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);
static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);
static PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);

#[global_allocator]
static ALLOCATOR: TrackingAllocator = TrackingAllocator;

fn record_allocation(size: usize) {
    ALLOCATION_COUNT.fetch_add(1, Ordering::Relaxed);
    ALLOCATED_BYTES.fetch_add(size, Ordering::Relaxed);
    let live = LIVE_BYTES.fetch_add(size, Ordering::Relaxed) + size;
    PEAK_BYTES.fetch_max(live, Ordering::Relaxed);
}

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            record_allocation(layout.size());
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            record_allocation(layout.size());
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        LIVE_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let resized = unsafe { System.realloc(pointer, layout, new_size) };
        if !resized.is_null() {
            ALLOCATION_COUNT.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(new_size, Ordering::Relaxed);
            if new_size >= layout.size() {
                let growth = new_size - layout.size();
                let live = LIVE_BYTES.fetch_add(growth, Ordering::Relaxed) + growth;
                PEAK_BYTES.fetch_max(live, Ordering::Relaxed);
            } else {
                LIVE_BYTES.fetch_sub(layout.size() - new_size, Ordering::Relaxed);
            }
        }
        resized
    }
}

fn generate_signal(sample_rate: u32, duration_secs: usize) -> Array1<Float> {
    Array1::from_shape_fn(sample_rate as usize * duration_secs, |index| {
        let time = index as Float / sample_rate as Float;
        0.3 * (2.0 * PI * 220.0 * time).sin()
            + 0.2 * (2.0 * PI * 660.0 * time).sin()
            + 0.1 * (2.0 * PI * 1_100.0 * time).sin()
    })
}

fn measure(label: &str, operation: impl FnOnce()) {
    let count = ALLOCATION_COUNT.load(Ordering::Relaxed);
    let bytes = ALLOCATED_BYTES.load(Ordering::Relaxed);
    let live = LIVE_BYTES.load(Ordering::Relaxed);
    PEAK_BYTES.store(live, Ordering::Relaxed);
    operation();
    println!(
        "{label}: allocations={}, allocated_bytes={}, peak_additional_bytes={}",
        ALLOCATION_COUNT.load(Ordering::Relaxed) - count,
        ALLOCATED_BYTES.load(Ordering::Relaxed) - bytes,
        PEAK_BYTES.load(Ordering::Relaxed).saturating_sub(live),
    );
}

fn main() {
    let fused = AnalysisConfig {
        features: Some(HashSet::from([
            "aggression".to_owned(),
            "embedding".to_owned(),
        ])),
        ..AnalysisConfig::default()
    };
    for sample_rate in [22_050, 48_000] {
        let signal = generate_signal(sample_rate, 30);
        let _ = aggression::analyze_signal(signal.view(), sample_rate).unwrap();
        let _ = analyze::analyze_signal(signal.view(), sample_rate, &fused).unwrap();
        measure(&format!("standalone/{sample_rate}hz-30s"), || {
            let _ = aggression::analyze_signal(signal.view(), sample_rate).unwrap();
        });
        measure(&format!("embedding+aggression/{sample_rate}hz-30s"), || {
            let _ = analyze::analyze_signal(signal.view(), sample_rate, &fused).unwrap();
        });
    }
}
