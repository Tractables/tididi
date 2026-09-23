use super::*;

mod support;

#[test]
fn validation_rejects_corrupt_traversal_views() {
    let original = Vtree::balanced(4);
    let corruptions: [fn(&mut TopoOrder); 8] = [
        |topo| topo.leaves[0] = topo.leaves[1],
        |topo| topo.internal[0] = topo.internal[1],
        |topo| topo.leaves.swap(0, 1),
        |topo| topo.internal.swap(0, 1),
        |topo| topo.leaves[0] = VtreeIdx(u32::MAX),
        |topo| topo.internal[0] = VtreeIdx(u32::MAX),
        |topo| { topo.leaves.pop(); },
        |topo| { topo.internal.push(topo.internal[0]); },
    ];
    for corrupt in corruptions {
        let mut vtree = original.clone();
        corrupt(&mut vtree.topo);
        assert!(matches!(vtree.validate(), Err(super::super::VtreeError::Invalid(_))));
    }
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
            topo.leaves.reverse();
            topo.internal.reverse();
        },
    ];
    for corrupt in corruptions {
        let mut vtree = original.clone();
        corrupt(&mut vtree.topo);
        assert!(vtree.validate().is_err());
    }
}
