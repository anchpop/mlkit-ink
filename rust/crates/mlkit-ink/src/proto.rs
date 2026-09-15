//! A protobuf wire-format reader, and nothing more.
//!
//! The recospec schema is only partly recovered: most of its fields still have
//! no known name or meaning (see `proto/recospec.proto`). A code generator
//! would force us to commit to a guess for every one of them. Walking the wire
//! format directly lets us name the fields we actually recovered, skip the
//! rest without discarding them, and keeps the crate dependency-free.

use crate::error::Result;

/// One wire value. Lifetimes borrow the caller's buffer; nothing is copied.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Value<'a> {
    Varint(u64),
    Fixed64(u64),
    Bytes(&'a [u8]),
    Fixed32(u32),
}

impl<'a> Value<'a> {
    pub fn as_u64(&self) -> Result<u64> {
        match self {
            Value::Varint(v) => Ok(*v),
            _ => bail!(Format, "expected a varint, found {self:?}"),
        }
    }

    pub fn as_bool(&self) -> Result<bool> {
        Ok(self.as_u64()? != 0)
    }

    /// float32, whether stored as `fixed32` or (never observed) a varint.
    pub fn as_f32(&self) -> Result<f32> {
        match self {
            Value::Fixed32(v) => Ok(f32::from_bits(*v)),
            _ => bail!(Format, "expected a fixed32 float, found {self:?}"),
        }
    }

    pub fn as_f64(&self) -> Result<f64> {
        match self {
            Value::Fixed64(v) => Ok(f64::from_bits(*v)),
            Value::Fixed32(v) => Ok(f32::from_bits(*v) as f64),
            _ => bail!(Format, "expected a float, found {self:?}"),
        }
    }

    pub fn as_bytes(&self) -> Result<&'a [u8]> {
        match self {
            Value::Bytes(b) => Ok(b),
            _ => bail!(Format, "expected a length-delimited field, found {self:?}"),
        }
    }

    pub fn as_str(&self) -> Result<&'a str> {
        core::str::from_utf8(self.as_bytes()?)
            .map_err(|_| err!(Format, "protobuf string field is not valid UTF-8"))
    }
}

/// Sequential reader over one message body.
#[derive(Debug, Clone)]
pub struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Reader { data, pos: 0 }
    }

    pub fn is_done(&self) -> bool {
        self.pos >= self.data.len()
    }

    fn varint(&mut self) -> Result<u64> {
        let mut value = 0u64;
        for shift in (0..70).step_by(7) {
            ensure!(self.pos < self.data.len(), Format, "truncated varint");
            let byte = self.data[self.pos];
            self.pos += 1;
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        bail!(Format, "varint longer than 10 bytes")
    }

    /// Next `(field number, value)`, or `None` at the end of the message.
    pub fn next_field(&mut self) -> Result<Option<(u32, Value<'a>)>> {
        if self.is_done() {
            return Ok(None);
        }
        let key = self.varint()?;
        let field = u32::try_from(key >> 3)
            .map_err(|_| err!(Format, "protobuf field number out of range"))?;
        ensure!(field != 0, Format, "protobuf field number 0");
        let value = match key & 7 {
            0 => Value::Varint(self.varint()?),
            1 => Value::Fixed64(u64::from_le_bytes(self.take(8)?.try_into().unwrap())),
            2 => {
                let len = usize::try_from(self.varint()?)
                    .map_err(|_| err!(Format, "length-delimited field too large"))?;
                Value::Bytes(self.take(len)?)
            }
            5 => Value::Fixed32(u32::from_le_bytes(self.take(4)?.try_into().unwrap())),
            other => bail!(Unsupported, "protobuf wire type {other}"),
        };
        Ok(Some((field, value)))
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| err!(Format, "protobuf length overflow"))?;
        ensure!(
            end <= self.data.len(),
            Format,
            "truncated protobuf field at byte {}",
            self.pos
        );
        let slice = &self.data[self.pos..end];
        self.pos = end;
        Ok(slice)
    }
}

/// Visit every field of a message in wire order.
pub fn for_each_field<'a>(
    data: &'a [u8],
    mut visit: impl FnMut(u32, Value<'a>) -> Result<()>,
) -> Result<()> {
    let mut reader = Reader::new(data);
    while let Some((field, value)) = reader.next_field()? {
        visit(field, value)?;
    }
    Ok(())
}
