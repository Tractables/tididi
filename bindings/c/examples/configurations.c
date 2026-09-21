// scenario: docs/scenarios.md#configurations
#include "example.h"

int main(void) {
    TididiVtree *vtree = NULL;
    check(tididi_vtree_balanced(4, &vtree));
    section("rules");
    // begin: rules
    enum { LOCAL = 1, REMOTE, ENCRYPTED, NOTIFICATIONS };
    const int64_t destination[] = {LOCAL, REMOTE};
    const int64_t encryption[] = {-REMOTE, ENCRYPTED};
    TididiCircuit *a = NULL, *b = NULL, *valid = NULL;
    check(tididi_clause(vtree, destination, 2, &a, NULL));
    check(tididi_clause(vtree, encryption, 2, &b, NULL));
    check(tididi_and(a, b, &valid, NULL));
    printf("Valid configurations: %" PRIu64 "\n", count(valid));
    // end: rules
    release(a); release(b);

    section("forced");
    // begin: forced
    TididiCircuit *remote = NULL, *selected = NULL, *rules_copy = copy(valid);
    check(tididi_literal(vtree, REMOTE, &remote, NULL));
    check(tididi_and(rules_copy, remote, &selected, NULL));
    printf("Valid remote configurations: %" PRIu64 "\n", count(selected));
    TididiLiterals *forced = NULL;
    check(tididi_implied_literals(selected, &forced, NULL));
    printf("Forced literals:");
    for (size_t i = 0; i < tididi_literals_len(forced); ++i)
        printf(" %" PRId64, tididi_literals_data(forced)[i]);
    printf("\n");
    // end: forced
    tididi_literals_free(forced);
    release(rules_copy); release(remote); release(selected);

    section("observe");
    // begin: observe
    TididiCounter *counter = NULL;
    TididiCircuit *counter_input = copy(valid);
    check(tididi_counter(counter_input, &counter));
    const int64_t remote_choice[] = {REMOTE};
    const int64_t quiet[] = {-NOTIFICATIONS};
    uint64_t remaining = 0;
    check(tididi_counter_observe(counter, remote_choice, 1));
    check(tididi_counter_model_count(counter, &remaining, NULL));
    printf("Remote: %" PRIu64 "\n", remaining);
    check(tididi_counter_observe(counter, quiet, 1));
    check(tididi_counter_model_count(counter, &remaining, NULL));
    printf("Remote, notifications off: %" PRIu64 "\n", remaining);
    check(tididi_counter_clear_all(counter));
    check(tididi_counter_model_count(counter, &remaining, NULL));
    printf("All choices open: %" PRIu64 "\n", remaining);
    // end: observe
    check(tididi_counter_free(counter));
    release(counter_input); release(valid); tididi_vtree_free(vtree);
    return 0;
}
