// scenario: docs/scenarios.md#vtrees
#include "example.h"

// begin: formula
static TididiCircuit *formula(TididiVtree *vtree) {
    TididiCircuit *x[4] = {NULL};
    for (int64_t i = 0; i < 4; ++i) check(tididi_literal(vtree, i + 1, &x[i], NULL));
    TididiCircuit *a = NULL, *b = NULL, *same_a = NULL, *same_b = NULL;
    check(tididi_xor(x[0], x[2], &a, NULL));
    check(tididi_xor(x[1], x[3], &b, NULL));
    check(tididi_negate(a, &same_a, NULL));
    check(tididi_negate(b, &same_b, NULL));
    TididiCircuit *joined = NULL, *result = NULL;
    check(tididi_and(same_a, same_b, &joined, NULL));
    check(tididi_minimize(joined, &result, NULL));
    for (size_t i = 0; i < 4; ++i) release(x[i]);
    release(a); release(b); release(same_a); release(same_b); release(joined);
    return result;
}
// end: formula

int main(void) {
    section("grouping");
    // begin: grouping
    const uint32_t orders[][4] = {{1,3,2,4}, {1,2,3,4}};
    const char *names[] = {"Related variables together", "Related variables separated"};
    for (size_t i = 0; i < 2; ++i) {
        TididiVtree *vtree = NULL;
        check(tididi_vtree_balanced_over(orders[i], 4, &vtree));
        TididiCircuit *f = formula(vtree);
        size_t nodes = 0, pairs = 0;
        check(tididi_size(f, &nodes, &pairs));
        printf("%s: %" PRIu64 " models, %zu pairs\n", names[i], count(f), pairs);
        release(f); tididi_vtree_free(vtree);
    }
    // end: grouping
    return 0;
}
