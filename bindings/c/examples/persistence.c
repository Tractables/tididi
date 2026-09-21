// scenario: docs/scenarios.md#persistence
#include "example.h"

int main(void) {
    TididiVtree *vtree = NULL;
    check(tididi_vtree_balanced(3, &vtree));
    TididiCircuit *destination = NULL, *encryption = NULL;
    const int64_t a[] = {1,2}, b[] = {-2,3};
    check(tididi_clause(vtree, a, 2, &destination, NULL));
    check(tididi_clause(vtree, b, 2, &encryption, NULL));
    // begin: save
    char *vtree_text = NULL;
    TididiBytes *first = NULL, *second = NULL;
    check(tididi_vtree_to_text(vtree, &vtree_text));
    check(tididi_to_bytes(destination, &first));
    check(tididi_to_bytes(encryption, &second));
    // end: save

    section("restore");
    // begin: restore
    TididiVtree *restored = NULL;
    TididiCircuit *left = NULL, *right = NULL, *rules = NULL;
    check(tididi_vtree_from_text(vtree_text, &restored));
    check(tididi_from_bytes(restored, tididi_bytes_data(first), tididi_bytes_len(first), &left));
    check(tididi_from_bytes(restored, tididi_bytes_data(second), tididi_bytes_len(second), &right));
    check(tididi_and(left, right, &rules, NULL));
    printf("Restored valid configurations: %" PRIu64 "\n", count(rules));
    // end: restore
    release(destination); release(encryption); release(left); release(right); release(rules);
    tididi_bytes_free(first); tididi_bytes_free(second); tididi_string_free(vtree_text);
    tididi_vtree_free(vtree); tididi_vtree_free(restored);
    return 0;
}
