//! The one error reading or writing a diagram returns.

/// What can go wrong moving a diagram between memory and bytes.
///
/// Two kinds, and the split is where the fault lies. [`IoError::Io`] is the
/// stream's: a file that would not open, a disk that filled, a reader that
/// ended early. [`IoError::Format`] is the data's: a diagram the format cannot
/// carry on the way out, or bytes that do not describe one on the way in.
///
/// [`IoError::Format`] carries its detail as a message string; a caller that
/// wants to branch on a malformed file matches the variant, not the message.
#[derive(Debug)]
#[non_exhaustive]
pub enum IoError {
    /// The underlying file or stream failed.
    Io(std::io::Error),
    /// The diagram and the format do not agree.
    ///
    /// Writing: the diagram has a marginal level, which stores per-node model
    /// counts rather than nodes and has no structural form to emit. Reading: a
    /// record is malformed, references a node that does not exist, or
    /// contradicts the vtree the caller supplied. The message names the record
    /// and what was expected.
    Format(String),
}

impl std::fmt::Display for IoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IoError::Io(e) => write!(f, "{e}"),
            IoError::Format(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for IoError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            IoError::Io(e) => Some(e),
            IoError::Format(_) => None,
        }
    }
}

impl From<std::io::Error> for IoError {
    fn from(e: std::io::Error) -> Self {
        IoError::Io(e)
    }
}
