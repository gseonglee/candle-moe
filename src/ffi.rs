use core::ffi::{c_int, c_long, c_void};

extern "C" {
    pub(crate) fn moe_sum(
        input: *const c_void,
        output: *const c_void,

        dtype: u32,
    );

    pub(crate) fn moe_wna16_gemm(
        input: *const c_void,
        output: *const c_void,
        b_qweight: *const c_void,
        b_scales: *const c_void,
        b_qzeros: *const c_void,
        topk_weights: *const c_void,
        sorted_token_ids: *const c_void,
        expert_ids: *const c_void,
        num_tokens_post_pad: *const c_void,

        top_k: c_long,
        BLOCK_SIZE_M: c_long,
        BLOCK_SIZE_N: c_long,
        BLOCK_SIZE_K: c_long,
        bit: c_long,

        dtype: u32,
    );

    pub(crate) fn topk_softmax(
        topk_weight: *const c_void,
        topk_indices: *const c_void,
        token_expert_indices: *const c_void,
        gating_output: *const c_void,

        dtype: u32,
    );
}
