mod ffi;

use candle::cuda_backend::cudarc::driver::DevicePtr;
use candle::{DType, Device, Result, Storage, Tensor};
use half::{bf16, f16};
use std::ffi::{c_int, c_long};

pub fn apply_topk_softmax_<
    T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
>(
    topk_weight: &Tensor,
    topk_indices: &Tensor,
    token_expert_indices: &Tensor,
    gating_output: &Tensor,
) -> Result<()> {
    let dtype = topk_weight.dtype();
    if topk_indices.dtype() != dtype
        || token_expert_indices.dtype() != dtype
        || gating_output.dtype() != dtype
    {
        candle::bail!("apply_topk_softmax expects all tensors to have the same dtype");
    }

    let (w, w_l) = topk_weight.storage_and_layout();
    let w = match &*w {
        Storage::Cuda(w) => w,
        _ => candle::bail!("topk_weight must be a cuda tensor"),
    };

    let (i, i_l) = topk_indices.storage_and_layout();
    let i = match &*i {
        Storage::Cuda(i) => i,
        _ => candle::bail!("topk_indices must be a cuda tensor"),
    };

    let (ei, ei_l) = token_expert_indices.storage_and_layout();
    let ei: &candle::CudaStorageei = match &*ei {
        Storage::Cuda(ei) => ei,
        _ => candle::bail!("token_expert_indices must be a cuda tensor"),
    };

    let (g, g_l) = gating_output.storage_and_layout();
    let g: &candle::CudaStorageei = match &*g {
        Storage::Cuda(g) => g,
        _ => candle::bail!("gating_output must be a cuda tensor"),
    };

    let w_rank = w_l.stride().len();
    let i_rank = i_l.stride().len();
    let ei_rank = ei_l.stride().len();
    let g_rank = g_l.stride().len();

    if w_rank != 2 || i_rank != 2 || ei_rank != 2 || g_rank != 2 {
        candle::bail!(
            "apply_topk_softmax_inplace expects input tensors of rank 2 (w: {w_l:?}, i: {i_l:?}, ei: {ei_l:?}, g: {g_l:?})"
        )
    }

    // Get cuda slices for all tensors
    let w = w.as_cuda_slice::<T>()?;
    let i = i.as_cuda_slice::<T>()?;
    let ei = ei.as_cuda_slice::<T>()?;
    let g = g.as_cuda_slice::<T>()?;

    // Get cuda views for all tensors
    let w = w.slice(w_l.start_offset()..);
    let i = i.slice(i_l.start_offset()..);
    let ei = ei.slice(ei_l.start_offset()..);
    let g = g.slice(g_l.start_offset()..);

    let (num_tokens, top_k) = w_l.shape().dims2()?;

    if (num_tokens, top_k) != i_l.shape().dims2()? {
        candle::bail!(
            "shape mismatch topk_indices {:?}, expected {:?}",
            i_l.shape(),
            (num_tokens, top_k)
        )
    }

    if (num_tokens, top_k) != ei_l.shape().dims2()? {
        candle::bail!(
            "shape mismatch token_expert_indices {:?}, expected {:?}",
            ei_l.shape(),
            (num_tokens, top_k)
        )
    }

    let weight_stride = w_l.stride()[0];
    let indices_stride = i_l.stride()[0];

    let weight_ptr = *w.device_ptr() as *const core::ffi::c_void;
    let indices_ptr = *i.device_ptr() as *const core::ffi::c_void;
    let expert_indices_ptr = *ei.device_ptr() as *const core::ffi::c_void;
    let gate_ptr = *g.device_ptr() as *const core::ffi::c_void;

    unsafe { ffi::topk_softmax(weight_ptr, indices_ptr, expert_indices_ptr, gate_ptr) }

    Ok(())
}

pub fn apply_topk_softmax_inplace(
    topk_weight: &Tensor,
    topk_indices: &Tensor,
    token_expert_indices: &Tensor,
    gating_output: &Tensor,
) -> Result<()> {
    match topk_weight.dtype() {
        DType::F16 => apply_topk_softmax_::<f16>(
            topk_weight,
            topk_indices,
            token_expert_indices,
            gating_output,
        ),
        DType::BF16 => apply_topk_softmax_::<bf16>(
            topk_weight,
            topk_indices,
            token_expert_indices,
            gating_output,
        ),
        DType::F32 => apply_topk_softmax_::<f32>(
            topk_weight,
            topk_indices,
            token_expert_indices,
            gating_output,
        ),
        dt => {
            candle::bail!("apply_topk_softmax is only supported for f32, f16 and bf16 ({dt:?})")
        }
    }
}
