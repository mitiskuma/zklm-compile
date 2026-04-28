pub mod sumcheck;
pub mod pcs;
#[allow(dead_code)]
pub mod basefold;
pub mod matmul;
pub mod elementwise;
pub mod lookup;
pub mod layernorm;
pub mod rmsnorm;
pub mod gelu;
pub mod silu;
pub mod sigmoid;
pub mod sigmoid_gate;
pub mod swiglu;
pub mod softmax;
pub mod attention;
pub mod weight_commitment;
/// P10-7: GDN recurrent-state sumcheck — DESIGN
/// SCAFFOLD ONLY. The proof STRUCTURE is defined; the prover/verifier
/// math lands in a focused crypto session, not autonomous /loop work.
/// See module doc for the full design + open soundness questions.
#[allow(dead_code)]
pub mod gdn_recurrence;
