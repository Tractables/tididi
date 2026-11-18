use super::*;

mod support;

#[test]
fn validation_rejects_an_order_with_an_internal_node_before_a_leaf() {
    let mut vtree = Vtree::balanced(4);
    let (leaf, internal) = (vtree.topo.order[3], vtree.topo.order[4]);
    assert!(vtree.node(leaf).is_leaf() && vtree.children(internal) == (VtreeIdx(0), VtreeIdx(1)));
    vtree.topo.order.swap(3, 4);
    vtree.topo.pos.swap(leaf.idx(), internal.idx());
    assert!(matches!(vtree.validate(), Err(super::super::VtreeError::Invalid(_))));
}

#[test]
fn validation_rejects_corrupt_order_and_inverse() {
    let original = Vtree::balanced(4);
    let corruptions: [fn(&mut TopoOrder); 5] = [
        |topo| topo.order[0] = VtreeIdx(u32::MAX),
        |topo| topo.order[0] = topo.order[1],
        |topo| { topo.pos.pop(); },
        |topo| topo.pos[0] = u32::MAX,
        |topo| {
            topo.order.reverse();
            for (pos, t) in topo.order.iter().enumerate() { topo.pos[t.idx()] = pos as u32; }
        },
    ];
    for corrupt in corruptions {
        let mut vtree = original.clone();
        corrupt(&mut vtree.topo);
        assert!(vtree.validate().is_err());
    }
}
