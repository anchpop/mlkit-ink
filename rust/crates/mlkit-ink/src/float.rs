//! Float math for a `no_std` crate.
//!
//! `sqrt`, `atan2`, `exp` and friends live in `std`, not `core` — `core` only
//! has the operations the CPU is guaranteed to provide. This trait supplies
//! them either way:
//!
//! * With the default `std` feature the inherent `std` methods win name
//!   resolution, so calls go to the **platform libm**. That is what the Python
//!   reference implementation was validated against, and it is what every
//!   target we actually ship to (desktop, wasm32, iOS, Android) uses.
//! * Without `std`, the `libm` feature routes the same calls to the pure-Rust
//!   implementation instead.
//!
//! Those two are not bit-identical in the last ulp, and this pipeline has
//! threshold comparisons that a last-ulp difference could in principle flip.
//! So `std` is the default and the supported configuration; `libm` exists for
//! genuinely bare-metal targets and is not validated against the oracle.

#![allow(dead_code)]

/// Transcendental and root operations `core` does not provide.
///
/// Import it in any module doing float math, gated so neither configuration
/// warns:
///
/// ```ignore
/// #[cfg(all(not(feature = "std"), not(test)))]
/// use crate::float::Float;
/// ```
///
/// The `not(test)` half is needed because the unit-test harness links `std`
/// regardless of the feature, so without it the import is dead under
/// `cargo test --no-default-features`.
///
/// Under `std` the inherent methods win name resolution and the import would
/// be dead; without `std` the trait is the only thing supplying them.
pub trait Float {
    fn sqrt(self) -> Self;
    fn abs(self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn ln_1p(self) -> Self;
    fn tanh(self) -> Self;
    fn atan2(self, other: Self) -> Self;
    fn hypot(self, other: Self) -> Self;
    fn floor(self) -> Self;
    fn round(self) -> Self;
}

#[cfg(not(feature = "std"))]
impl Float for f32 {
    #[inline]
    fn sqrt(self) -> Self {
        libm::sqrtf(self)
    }
    #[inline]
    fn abs(self) -> Self {
        libm::fabsf(self)
    }
    #[inline]
    fn exp(self) -> Self {
        libm::expf(self)
    }
    #[inline]
    fn ln(self) -> Self {
        libm::logf(self)
    }
    #[inline]
    fn ln_1p(self) -> Self {
        libm::log1pf(self)
    }
    #[inline]
    fn tanh(self) -> Self {
        libm::tanhf(self)
    }
    #[inline]
    fn floor(self) -> Self {
        libm::floorf(self)
    }
    #[inline]
    fn round(self) -> Self {
        libm::roundf(self)
    }
    #[inline]
    fn atan2(self, other: Self) -> Self {
        libm::atan2f(self, other)
    }
    #[inline]
    fn hypot(self, other: Self) -> Self {
        libm::hypotf(self, other)
    }
}

#[cfg(not(feature = "std"))]
impl Float for f64 {
    #[inline]
    fn sqrt(self) -> Self {
        libm::sqrt(self)
    }
    #[inline]
    fn abs(self) -> Self {
        libm::fabs(self)
    }
    #[inline]
    fn exp(self) -> Self {
        libm::exp(self)
    }
    #[inline]
    fn ln(self) -> Self {
        libm::log(self)
    }
    #[inline]
    fn ln_1p(self) -> Self {
        libm::log1p(self)
    }
    #[inline]
    fn tanh(self) -> Self {
        libm::tanh(self)
    }
    #[inline]
    fn floor(self) -> Self {
        libm::floor(self)
    }
    #[inline]
    fn round(self) -> Self {
        libm::round(self)
    }
    #[inline]
    fn atan2(self, other: Self) -> Self {
        libm::atan2(self, other)
    }
    #[inline]
    fn hypot(self, other: Self) -> Self {
        libm::hypot(self, other)
    }
}

// Under `std` the trait exists only so that `use crate::float::Float;` compiles
// and stays meaningful; every call resolves to the inherent method first.
#[cfg(feature = "std")]
macro_rules! impl_float_std {
    ($ty:ty) => {
        impl Float for $ty {
            #[inline]
            fn sqrt(self) -> Self {
                <$ty>::sqrt(self)
            }
            #[inline]
            fn abs(self) -> Self {
                <$ty>::abs(self)
            }
            #[inline]
            fn exp(self) -> Self {
                <$ty>::exp(self)
            }
            #[inline]
            fn ln(self) -> Self {
                <$ty>::ln(self)
            }
            #[inline]
            fn ln_1p(self) -> Self {
                <$ty>::ln_1p(self)
            }
            #[inline]
            fn tanh(self) -> Self {
                <$ty>::tanh(self)
            }
            #[inline]
            fn atan2(self, other: Self) -> Self {
                <$ty>::atan2(self, other)
            }
            #[inline]
            fn hypot(self, other: Self) -> Self {
                <$ty>::hypot(self, other)
            }
            #[inline]
            fn floor(self) -> Self {
                <$ty>::floor(self)
            }
            #[inline]
            fn round(self) -> Self {
                <$ty>::round(self)
            }
        }
    };
}

#[cfg(feature = "std")]
impl_float_std!(f32);
#[cfg(feature = "std")]
impl_float_std!(f64);
