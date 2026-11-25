mod binary;
mod writers;
mod readers;
mod dot;
mod home;

use std::io::{BufRead, Read};
use std::sync::Arc;

use crate::diagram::Tdd;
use crate::io::IoError;
use crate::test_helpers::check::check_trimmed;
use crate::vtree::{Home, Vtree};

/// [`crate::io::read_tdd`], and for a file it reads over a root vtree, the
/// same file read onto a home of a copy of that vtree: trimmed, and the same
/// function. Every file these tests read takes both paths.
pub(super) fn read_tdd<R: BufRead>(r: &mut R, vtree: &crate::vtree::Home) -> Result<Tdd, IoError> {
    let mut bytes = Vec::new();
    r.read_to_end(&mut bytes)?;
    let read = crate::io::read_tdd(&mut bytes.as_slice(), vtree);
    if let Ok(f) = &read {
        also_on_home(f, vtree, |home| crate::io::read_tdd(&mut bytes.as_slice(), &crate::vtree::Home::new(std::sync::Arc::clone(home))));
    }
    read
}

/// [`crate::io::read_tdd_binary`], with the check of [`read_tdd`] on the
/// bytes the read took.
pub(super) fn read_tdd_binary<R: Read>(r: &mut R, vtree: &crate::vtree::Home) -> Result<Tdd, IoError> {
    let mut tee = Tee { inner: r, seen: Vec::new() };
    let read = crate::io::read_tdd_binary(&mut tee, vtree);
    if let Ok(f) = &read {
        let bytes = tee.seen;
        also_on_home(f, vtree, |home| crate::io::read_tdd_binary(&mut bytes.as_slice(), &crate::vtree::Home::new(std::sync::Arc::clone(home))));
    }
    read
}

struct Tee<'a, R> {
    inner: &'a mut R,
    seen: Vec<u8>,
}

impl<R: Read> Read for Tee<'_, R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.seen.extend_from_slice(&buf[..n]);
        Ok(n)
    }
}

fn also_on_home(f: &Tdd, vtree: &crate::vtree::Home, read: impl Fn(&Arc<Vtree>) -> Result<Tdd, IoError>) {
    if vtree.home().is_some() || vtree.num_nodes() == 0 {
        return;
    }
    let home = Home::new(Vtree::clone(vtree));
    let g = match read(&home) {
        Ok(g) => g,
        // The binary reader looks for a repeated pair only on a home, where
        // the trim would count it once.
        Err(IoError::Format(m)) if m.contains("twice") => return,
        Err(e) => panic!("read over the root, refused on a home: {e}"),
    };
    check_trimmed(&g, false).unwrap_or_else(|e| panic!("read onto a home: {e}"));
    assert!(g.vtree().is_on(&home));
    // Counts read from a file are trusted, and a damaged file's need not
    // agree with the diagram the trim leaves.
    if !f.has_level_counts() {
        assert_eq!(g.model_count().unwrap(), f.model_count().unwrap());
    }
    let vars: Vec<usize> = vtree.leaf_bottomup().map(|(_, v)| v.idx()).collect();
    let top = vars.iter().copied().max().unwrap_or(0);
    let mut rng = crate::vtree::rng::Lcg::new(vars.len() as u64);
    for k in 0..256u64.min(1 << vars.len().min(16)) {
        let mut asn = vec![false; top + 1];
        for (i, &v) in vars.iter().enumerate() {
            asn[v] = if vars.len() <= 8 { k >> i & 1 == 1 } else { rng.coin() };
        }
        let on_home = if g.vtree().num_nodes() == 0 { g.is_one() } else { crate::test_helpers::eval(&g, &asn) };
        assert_eq!(on_home, crate::test_helpers::eval(f, &asn), "assignment {asn:?}");
    }
}
