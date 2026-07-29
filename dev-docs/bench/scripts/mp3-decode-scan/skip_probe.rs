use std::fs::File;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

fn main() {
    for path in std::env::args().skip(1) {
        match probe_one(&path) {
            Ok((ok, skipped, secs)) => println!("{path}\t{ok}\t{skipped}\t\t{secs:.2}"),
            Err(e) => println!("{path}\t0\t0\tFATAL:{e}\t0.00"),
        }
    }
}

fn probe_one(path: &str) -> Result<(u64, u64, f64), String> {
    let path = path.to_string();
    let file = File::open(&path).map_err(|e| format!("open:{e}"))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = std::path::Path::new(&path).extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|e| format!("probe:{e}"))?;
    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
        .ok_or("no track")?
        .clone();
    let track_id = track.id;
    let sr = track.codec_params.sample_rate.ok_or("no sr")?;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| format!("codec:{e}"))?;

    let mut sample_buf: Option<SampleBuffer<f32>> = None;
    let mut total_frames: u64 = 0;
    let mut ok_packets: u64 = 0;
    let mut skipped: u64 = 0;
    let mut first_errors: Vec<String> = Vec::new();
    let mut reset_required: u64 = 0;

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(Error::IoError(ref e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(format!("demux:{e}")),
        };
        if packet.track_id() != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(decoded) => {
                ok_packets += 1;
                let spec = *decoded.spec();
                let cap = decoded.capacity();
                if sample_buf.is_none() || sample_buf.as_ref().unwrap().capacity() < cap {
                    sample_buf = Some(SampleBuffer::<f32>::new(cap as u64, spec));
                }
                let buf = sample_buf.as_mut().unwrap();
                buf.copy_interleaved_ref(decoded);
                total_frames += (buf.samples().len() / spec.channels.count()) as u64;
            }
            Err(Error::DecodeError(msg)) => {
                skipped += 1;
                if first_errors.len() < 5 {
                    first_errors.push(format!("packet#{} : {msg}", ok_packets + skipped));
                }
            }
            Err(Error::ResetRequired) => {
                reset_required += 1;
                decoder = symphonia::default::get_codecs()
                    .make(&track.codec_params, &DecoderOptions::default())
                    .map_err(|e| format!("codec:{e}"))?;
            }
            Err(e) => return Err(format!("decode-fatal after {ok_packets}:{e}")),
        }
    }
    let _ = (reset_required, first_errors);
    Ok((ok_packets, skipped, total_frames as f64 / sr as f64))
}
