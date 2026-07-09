//! Decode-loop primitives shared by every causal model: on-device argmax and
//! a top-k probe. The full engine (sampling, streaming, interrupts) is P5.

use burn::tensor::{Tensor, backend::Backend};

/// Greedy next-token id from `[1, vocab]` logits. The argmax runs
/// **on-device** and only the single winning index is synced back — vs.
/// copying a whole ~150k-logit vector to the CPU every decode step.
pub fn argmax_id<B: Backend>(logits: Tensor<B, 2>) -> Result<u32, String> {
    debug_assert!(logits.dims()[0] == 1, "argmax_id expects [1, vocab] logits");
    let data = logits
        .argmax(1)
        .into_data()
        .convert::<i64>()
        .to_vec::<i64>()
        .map_err(|e| format!("argmax readback: {e:?}"))?;
    debug_assert!(data.len() == 1, "argmax over [1, vocab] must yield one id");
    let id = data.first().copied().ok_or("argmax returned no data")?;
    Ok(id as u32)
}

/// Indices of the `k` largest values, descending (the parity probe's top-k).
#[must_use]
pub fn top_k_ids(v: &[f32], k: usize) -> Vec<u32> {
    assert!(!v.is_empty(), "top_k_ids: empty logits");
    assert!(k >= 1, "top_k_ids: k must be >= 1");
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_unstable_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap_or(std::cmp::Ordering::Equal));
    idx.into_iter().take(k).map(|i| i as u32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Cpu;
    use burn::tensor::Tensor;

    #[test]
    fn argmax_id_finds_the_peak() {
        let device = burn::tensor::Device::<Cpu>::default();
        let logits = Tensor::<Cpu, 1>::from_floats([0.1, -2.0, 7.5, 3.0], &device).reshape([1, 4]);
        assert_eq!(argmax_id(logits).unwrap(), 2);
    }

    #[test]
    fn top_k_ids_orders_descending() {
        let v = [0.1f32, 5.0, -2.0, 3.0];
        assert_eq!(top_k_ids(&v, 3), vec![1, 3, 0]);
    }
}
