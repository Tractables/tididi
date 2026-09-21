// scenario: docs/scenarios.md#statistics
#include "example.h"

int main(void) {
    TididiVtree *vtree = NULL;
    check(tididi_vtree_balanced(4, &vtree));
    TididiCircuit *x = NULL, *y = NULL, *f = NULL;
    check(tididi_literal(vtree, 1, &x, NULL));
    check(tididi_literal(vtree, 2, &y, NULL));
    check(tididi_xor(x, y, &f, NULL));
    section("sizes");
    // begin: sizes
    size_t nodes = 0, pairs = 0;
    check(tididi_size(f, &nodes, &pairs));
    printf("Models: %" PRIu64 "; stored nodes: %zu; pairs: %zu\n", count(f), nodes, pairs);
    TididiNodeSizes *snapshot = NULL;
    check(tididi_node_sizes(f, &snapshot));
    size_t largest = 0;
    for (size_t i = 0; i < tididi_node_sizes_len(snapshot); ++i) {
        const TididiNodeSize *row = &tididi_node_sizes_data(snapshot)[i];
        if (row->pairs > largest) largest = row->pairs;
    }
    printf("Largest node: %zu pairs\n", largest);
    tididi_node_sizes_free(snapshot);
    // end: sizes
    release(x); release(y); release(f); tididi_vtree_free(vtree);
    return 0;
}
