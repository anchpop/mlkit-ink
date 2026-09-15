//! One error type for the whole crate.
//!
//! Every artifact we read is Google's, shipped, and byte-exact; a parse failure
//! means our understanding of the format is wrong, not that the user did
//! something wrong. So errors carry enough context to point at the offending
//! offset or field rather than just saying "invalid".

use alloc::string::String;
use core::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// A shipped binary did not match the format recovered in SPEC.md.
    Format(String),
    /// The pipeline was asked for something the recovered format cannot express.
    Unsupported(String),
    /// Caller-supplied ink or parameters are inconsistent.
    Invalid(String),
    /// No model pack covers the requested BCP-47 tag.
    NoSuchLanguage(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Format(m) => write!(f, "malformed model artifact: {m}"),
            Error::Unsupported(m) => write!(f, "unsupported: {m}"),
            Error::Invalid(m) => write!(f, "invalid input: {m}"),
            Error::NoSuchLanguage(t) => write!(f, "no Digital Ink model for language tag {t:?}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

pub type Result<T> = core::result::Result<T, Error>;

/// `format!`-style constructors, so call sites stay one line.
macro_rules! err {
    ($variant:ident, $($arg:tt)*) => {
        $crate::error::Error::$variant(alloc::format!($($arg)*))
    };
}

macro_rules! bail {
    ($variant:ident, $($arg:tt)*) => {
        return Err(err!($variant, $($arg)*))
    };
}

macro_rules! ensure {
    ($cond:expr, $variant:ident, $($arg:tt)*) => {
        if !$cond {
            bail!($variant, $($arg)*);
        }
    };
}
