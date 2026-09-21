// scenario: docs/scenarios.md#execution
#include "example.h"

int main(void) {
    TididiVtree *vtree = NULL;
    check(tididi_vtree_balanced(4, &vtree));
    TididiCircuit *destination = NULL, *encrypted = NULL;
    const int64_t destinations[] = {1,2};
    check(tididi_clause(vtree, destinations, 2, &destination, NULL));
    check(tididi_literal(vtree, 3, &encrypted, NULL));
    section("retry");
    // begin: retry
    TididiLimits limits = tididi_limits_default();
    limits.memory_bytes = 0;
    TididiCircuit *a = copy(destination), *b = copy(encrypted), *result = NULL;
    TididiError *error = tididi_and(a, b, &result, &limits);
    if (error) {
        if (tididi_error_code(error) != TIDIDI_ERROR_CODE_MEMORY_LIMIT) check(error);
        puts("Operation exceeded its memory budget; retrying with fresh copies.");
        tididi_error_free(error);
        release(a); release(b);
        a = copy(destination); b = copy(encrypted);
        check(tididi_and(a, b, &result, NULL));
    }
    printf("Encrypted configurations: %" PRIu64 "\n", count(result));
    // end: retry
    release(a); release(b); release(result); release(destination); release(encrypted);
    check(tididi_vtree_clear_scratch(vtree)); tididi_vtree_free(vtree);
    return 0;
}
