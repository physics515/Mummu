//! The model zoo: from-scratch architectures on the shared `nn` blocks, all
//! generic over `B: Backend`, all config-driven (hyperparameters come from the
//! checkpoint's `config.json`, never hardcoded).

pub mod lfm2;
pub mod qwen2;
