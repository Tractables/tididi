// scenario: docs/scenarios.md#counting-choices
#include "example.h"

int main(void) {
    section("rules");
    // begin: rules
    enum { LOCAL = 1, REMOTE, ENCRYPTED, NOTIFICATIONS };
    TididiVtree *vtree = NULL;
    check(tididi_vtree_balanced(4, &vtree));
    const int64_t destinations[] = {LOCAL, REMOTE}, encryption[] = {-REMOTE, ENCRYPTED};
    TididiCircuit *a = NULL, *b = NULL, *rules = NULL;
    check(tididi_clause(vtree, destinations, 2, &a, NULL));
    check(tididi_clause(vtree, encryption, 2, &b, NULL));
    check(tididi_and(a, b, &rules, NULL));
    printf("Valid configurations: %" PRIu64 "\n", count(rules));
    // end: rules
    release(a); release(b);

    section("observe");
    // begin: observe
    const int64_t remote_choice[] = {REMOTE};
    TididiCircuit *remote = NULL, *selected = NULL, *rules_copy = copy(rules);
    check(tididi_literal(vtree, REMOTE, &remote, NULL));
    check(tididi_and(rules_copy, remote, &selected, NULL));
    printf("With remote backups: %" PRIu64 "\n", count(selected));
    TididiCounter *counter = NULL;
    TididiCircuit *counter_input = copy(rules);
    check(tididi_counter(counter_input, &counter));
    check(tididi_counter_observe(counter, remote_choice, 1));
    uint64_t matching = 0;
    check(tididi_counter_model_count(counter, &matching, NULL));
    printf("Observed remote backups: %" PRIu64 "\n", matching);
    // end: observe
    release(remote); release(rules_copy); release(selected); release(counter_input);
    check(tididi_counter_free(counter));

    section("substitute");
    // begin: substitute
    TididiCircuit *input = copy(rules), *residual = NULL;
    check(tididi_condition(input, remote_choice, 1, &residual, NULL));
    printf("After substituting remote = true: %" PRIu64 "\n", count(residual));
    const uint32_t remaining[] = {LOCAL, ENCRYPTED, NOTIFICATIONS};
    uint64_t distinct = 0;
    check(tididi_projected_model_count(residual, remaining, 3, &distinct, NULL));
    printf("Distinct remaining choices: %" PRIu64 "\n", distinct);
    // end: substitute
    release(input); release(residual);

    section("project");
    // begin: project
    const uint32_t keep[] = {LOCAL, REMOTE}, eliminate[] = {ENCRYPTED, NOTIFICATIONS};
    check(tididi_projected_model_count(rules, keep, 2, &distinct, NULL));
    printf("Valid destination choices: %" PRIu64 "\n", distinct);
    TididiCircuit *original = copy(rules), *destination_rule = NULL;
    check(tididi_exists(original, eliminate, 2, &destination_rule, NULL));
    printf("Destination rule over the full vtree: %" PRIu64 "\n", count(destination_rule));
    check(tididi_projected_model_count(destination_rule, keep, 2, &distinct, NULL));
    printf("Destination rule projected: %" PRIu64 "\n", distinct);
    // end: project
    release(original); release(destination_rule); release(rules); tididi_vtree_free(vtree);
    return 0;
}
