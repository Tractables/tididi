//! The `.vtree` text format, the format of the SDD package: nodes appear
//! children before parents, with 1-indexed variable ids. Both directions live
//! together: what one writes the other has to accept.

use super::{VarId, Vtree, VtreeError, VtreeIdx, VtreeNode};

impl Vtree {
    /// Parse the `.vtree` text format.
    ///
    /// Format: `vtree N` header, then N lines of `L <id> <var_1indexed>` or
    /// `I <id> <left> <right>`. Node ids are `0..N`, in any order; variables
    /// are 1-based. The last node listed is the root. Blank lines are skipped;
    /// comment lines are not accepted.
    ///
    /// # Errors
    ///
    /// [`VtreeError::Text`] describing what is wrong with `s`: a missing or
    /// invalid header, a malformed node line, an unparseable id, a node or
    /// variable id outside the range the header declares, a variable carried by
    /// two leaves, a duplicate node id, a node-record count different from the
    /// header, or node lines that do not describe a single tree.
    ///
    /// ```
    /// use tididi::vtree::{Vtree, VtreeError};
    ///
    /// let text = "vtree 3\nL 0 1\nL 1 2\nI 2 0 1\n";
    /// let vtree = Vtree::from_text(text).unwrap();
    /// assert_eq!(vtree.num_leaves(), 2);
    /// assert_eq!(vtree.to_text(), text);
    ///
    /// // A node line naming a child that was never declared is refused.
    /// match Vtree::from_text("vtree 2\nL 0 1\nI 1 0 7\n") {
    ///     Ok(_) => unreachable!(),
    ///     Err(VtreeError::Text(msg)) => assert!(!msg.is_empty()),
    ///     Err(other) => unreachable!("{other}"),
    /// }
    /// ```
    pub fn from_text(s: &str) -> Result<Self, VtreeError> {
        let vtree = Self::parse_vtree_text(s).map_err(VtreeError::Text)?;
        debug_assert_eq!(vtree.validate(), Ok(()));
        Ok(vtree)
    }

    /// The parse itself, reporting a plain sentence.
    fn parse_vtree_text(s: &str) -> Result<Self, String> {
        let mut lines = s.lines();
        let n = parse_header(lines.next().ok_or("empty vtree file")?)?;

        let records = lines.clone().filter(|line| !line.trim().is_empty()).count();
        if records != n {
            return Err(format!("header declares {n} nodes but the file contains {records} node records"));
        }
        let mut nodes = vec![None; n];
        let mut num_vars: u32 = 0;
        let mut last_id = 0usize;
        // A vtree carries each variable on exactly one leaf. A file naming one
        // twice is caught here rather than left to the leaf-count assertion a
        // consumer of the tree eventually trips over.
        let mut leaf_of_var: std::collections::HashMap<u32, usize> =
            std::collections::HashMap::new();

        for line in lines {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.split_whitespace().collect();
            let (id, node) = match parts[0] {
                "L" => parse_leaf_line(&parts, line, n, &mut num_vars, &mut leaf_of_var)?,
                "I" => parse_internal_line(&parts, line, n)?,
                _ => return Err(format!("unknown line type: {}", line)),
            };
            if nodes[id].is_some() { return Err(format!("duplicate node id {id}")); }
            nodes[id] = Some(node);
            last_id = id;
        }

        let nodes: Vec<VtreeNode> = nodes
            .into_iter()
            .enumerate()
            .map(|(i, n)| n.ok_or_else(|| format!("missing node {}", i)))
            .collect::<Result<_, _>>()?;

        let root = VtreeIdx(last_id as u32);
        Self::from_nodes(nodes, root, num_vars).map_err(|e| match e {
            VtreeError::Invalid(msg) | VtreeError::Text(msg) => msg,
            other => other.to_string(),
        })
    }

    /// Serialize this vtree in the `.vtree` text format.
    ///
    /// Nodes appear bottom-up (children before parents), with 1-indexed
    /// variable ids.
    ///
    /// The id printed for a node is its position in [`Vtree::bottomup`], which
    /// the format requires to precede its parent's. On a tree that has been
    /// rotated that position is no longer the node's index, so a [`VtreeIdx`]
    /// does not survive the round trip; the tree does. Equal text means equal
    /// tree, but not conversely once a tree has been rotated — compare trees
    /// with [`Vtree::same_tree`].
    pub fn to_text(&self) -> String {
        let n = self.num_nodes();
        let mut out = format!("vtree {}\n", n);
        for idx in self.bottomup() {
            let id = self.topo_pos(idx);
            match self.node(idx) {
                VtreeNode::Leaf { var, .. } => {
                    out.push_str(&format!("L {} {}\n", id, var.0 + 1));
                }
                VtreeNode::Internal { left, right, .. } => {
                    out.push_str(&format!(
                        "I {} {} {}\n",
                        id,
                        self.topo_pos(*left),
                        self.topo_pos(*right)
                    ));
                }
            }
        }
        out
    }
}

/// The declared node count from the `vtree N` header line.
fn parse_header(header: &str) -> Result<usize, String> {
    let n: usize = header
        .strip_prefix("vtree ")
        .ok_or("missing 'vtree N' header")?
        .trim()
        .parse()
        .map_err(|_| "invalid node count in header")?;
    if n == 0 {
        return Err("header declares 0 nodes; a vtree has at least one".to_string());
    }
    Ok(n)
}

/// Every id a node line names — the node's own, and an internal node's two
/// children — has to address a node the header declared.
fn check_id(what: &str, id: usize, n: usize) -> Result<(), String> {
    if id < n {
        Ok(())
    } else {
        Err(format!(
            "{what} {id} is out of range for the {n} nodes the header declares"
        ))
    }
}

/// An `L <id> <var_1indexed>` line, recording the variable so a second leaf
/// naming it is rejected.
fn parse_leaf_line(
    parts: &[&str],
    line: &str,
    n: usize,
    num_vars: &mut u32,
    leaf_of_var: &mut std::collections::HashMap<u32, usize>,
) -> Result<(usize, VtreeNode), String> {
    if parts.len() != 3 {
        return Err(format!("bad leaf line: {}", line));
    }
    let id: usize = parts[1]
        .parse()
        .map_err(|_| format!("bad id: {}", parts[1]))?;
    check_id("node id", id, n)?;
    let var_1: u32 = parts[2]
        .parse()
        .map_err(|_| format!("bad var: {}", parts[2]))?;
    if var_1 == 0 {
        return Err(format!(
            "leaf {id} names variable 0; vtree variables are 1-based"
        ));
    }
    if let Some(first) = leaf_of_var.insert(var_1, id) {
        return Err(format!(
            "leaves {first} and {id} both name variable {var_1}; a vtree carries \
             each variable on exactly one leaf"
        ));
    }
    let var = VarId(var_1 - 1); // `.vtree` ids are 1-indexed
    *num_vars = (*num_vars).max(var_1);
    Ok((id, VtreeNode::Leaf { var, parent: None }))
}

/// An `I <id> <left> <right>` line.
fn parse_internal_line(parts: &[&str], line: &str, n: usize) -> Result<(usize, VtreeNode), String> {
    if parts.len() != 4 {
        return Err(format!("bad internal line: {}", line));
    }
    let id: usize = parts[1]
        .parse()
        .map_err(|_| format!("bad id: {}", parts[1]))?;
    check_id("node id", id, n)?;
    let left: u32 = parts[2]
        .parse()
        .map_err(|_| format!("bad left: {}", parts[2]))?;
    check_id("left child", left as usize, n)?;
    let right: u32 = parts[3]
        .parse()
        .map_err(|_| format!("bad right: {}", parts[3]))?;
    check_id("right child", right as usize, n)?;
    Ok((
        id,
        VtreeNode::Internal {
            left: VtreeIdx(left),
            right: VtreeIdx(right),
            parent: None,
        },
    ))
}

/// The `.vtree` text format, so `vtree.to_string()` writes it.
impl std::fmt::Display for Vtree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.to_text())
    }
}

/// The `.vtree` text format, so `text.parse::<Vtree>()` reads it.
impl std::str::FromStr for Vtree {
    type Err = VtreeError;

    fn from_str(s: &str) -> Result<Self, VtreeError> {
        Vtree::from_text(s)
    }
}
