mod field;
mod proving;
#[cfg(feature = "metal_gpu")]
mod gpu;
mod pipeline;
mod protocol;
mod server;
mod transformer;
mod verification;

use std::io::{self, BufReader, BufWriter, Read};
use std::time::Instant;

use crate::pipeline::{run_gpt2_mode, run_mlp_mode};
use crate::verification::run_verify_mode;
use crate::server::run_server_mode;
use crate::protocol::{ProveRequest, parse_binary};

fn main() {
    // Check CLI args first
    let args: Vec<String> = std::env::args().collect();

    // --gpu flag enables Metal GPU acceleration at runtime
    #[cfg(feature = "metal_gpu")]
    if args.iter().any(|a| a == "--gpu") {
        crate::proving::sumcheck::set_gpu_enabled(true);
        eprintln!("  Metal GPU acceleration: ON");
    }

    // --fast flag opts INTO base-field proving (~31-bit challenges).
    // Default is extension-field (~124-bit challenges) for funding-grade soundness.
    // SOUNDNESS (M1, Option C): switch is documented as opt-in to base mode.
    if args.iter().any(|a| a == "--fast") {
        crate::proving::sumcheck::set_fast_mode(true);
        eprintln!("  Prover mode: --fast (base-field, ~31-bit challenges)");
    } else {
        eprintln!("  Prover mode: default (extension-field, ~124-bit challenges)");
    }

    if args.len() >= 3 && args[1] == "--verify" {
        run_verify_mode(&args[2]);
        return;
    }

    // Peek at first byte: 0x00 = binary protocol, 0x01 = server mode, otherwise JSON.
    let mut stdin = BufReader::new(io::stdin());
    let mut stdout = BufWriter::new(io::stdout());

    let mut first_byte = [0u8; 1];
    match stdin.read_exact(&mut first_byte) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return,
        Err(e) => panic!("IO error: {}", e),
    }

    if first_byte[0] == 0x01 {
        // Server mode
        eprintln!("  entering server mode");
        run_server_mode(&mut stdin, &mut stdout);
        return;
    }

    // One-shot mode: read remaining stdin
    let mut raw = vec![first_byte[0]];
    stdin.read_to_end(&mut raw).unwrap();

    let t_parse = Instant::now();
    let req: ProveRequest = if raw[0] == 0x00 {
        let req = parse_binary(&raw[1..]);
        eprintln!(
            "  binary parse: {:.1}ms ({:.1}MB)",
            t_parse.elapsed().as_secs_f64() * 1000.0,
            raw.len() as f64 / 1e6
        );
        req
    } else {
        let input = match String::from_utf8(raw) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Invalid UTF-8 in input: {}", e);
                std::process::exit(1);
            }
        };
        let req: ProveRequest = match serde_json::from_str(&input) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Failed to parse JSON: {}", e);
                std::process::exit(1);
            }
        };
        eprintln!(
            "  json parse: {:.1}ms ({:.1}MB)",
            t_parse.elapsed().as_secs_f64() * 1000.0,
            input.len() as f64 / 1e6
        );
        req
    };

    match req.mode.as_str() {
        "mlp" => run_mlp_mode(req),
        "gpt2" => run_gpt2_mode(req),
        other => {
            eprintln!("Unknown mode: {}. Supported: mlp, gpt2", other);
            std::process::exit(1);
        }
    }
}
