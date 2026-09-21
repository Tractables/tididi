// scenario: docs/scenarios.md#tables
#include "example.h"

int main(void) {
    TididiVtree *vtree = NULL;
    check(tididi_vtree_balanced(3, &vtree));
    section("rows");
    // begin: rows
    enum { READ = 1, WRITE, SHARE };
    const uint32_t columns[] = {READ, WRITE, SHARE};
    const uint8_t rows[][3] = {{1,0,0}, {1,1,0}, {1,0,1}, {1,1,0}};
    TididiCircuit *table = NULL;
    check(tididi_from_models(vtree, columns, 3, &rows[0][0], 4, &table, NULL));
    printf("Distinct rows: %" PRIu64 "\n", count(table));
    // end: rows

    section("update");
    // begin: update
    const int64_t add[] = {READ, WRITE, SHARE}, remove[] = {READ, WRITE, -SHARE};
    const TididiCube insertions[] = {{add, 3}}, deletions[] = {{remove, 3}};
    TididiCircuit *updated = NULL;
    check(tididi_update(table, insertions, 1, deletions, 1, &updated, NULL));
    printf("After updating: %" PRIu64 "\n", count(updated));
    // end: update

    section("filter");
    // begin: filter
    TididiCircuit *share = NULL, *selected = NULL;
    check(tididi_literal(vtree, SHARE, &share, NULL));
    check(tididi_and(updated, share, &selected, NULL));
    printf("Rows allowing sharing: %" PRIu64 "\n", count(selected));
    // end: filter
    release(table); release(updated); release(share); release(selected);
    tididi_vtree_free(vtree);
    return 0;
}
