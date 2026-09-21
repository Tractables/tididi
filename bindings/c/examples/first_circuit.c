// scenario: docs/scenarios.md#first-circuit
#include "example.h"

int main(void) {
    section("build");
    // begin: build
    TididiVtree *vtree = NULL;
    TididiCircuit *x = NULL, *y = NULL, *z = NULL;
    TididiCircuit *xy = NULL, *formula = NULL;
    check(tididi_vtree_balanced(3, &vtree));
    check(tididi_literal(vtree, 1, &x, NULL));
    check(tididi_literal(vtree, 2, &y, NULL));
    check(tididi_literal(vtree, 3, &z, NULL));
    check(tididi_and(x, y, &xy, NULL));
    check(tididi_or(xy, z, &formula, NULL));
    printf("Satisfying assignments: %" PRIu64 "\n", count(formula));
    // end: build

    section("copy");
    // begin: copy
    TididiCircuit *saved = copy(formula), *complement = NULL;
    check(tididi_negate(saved, &complement, NULL));
    printf("Original: %" PRIu64 "; complement: %" PRIu64 "\n",
           count(formula), count(complement));
    // end: copy

    // begin: cleanup
    release(x); release(y); release(z); release(xy);
    release(formula); release(saved); release(complement);
    tididi_vtree_free(vtree);
    // end: cleanup
    return 0;
}
