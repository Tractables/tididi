// scenario: docs/scenarios.md#reachability
#include "example.h"

// begin: state
static TididiCircuit *state(const TididiVtree *vtree, const uint32_t vars[16], size_t node) {
    int64_t indicators[16];
    for (size_t i = 0; i < 16; ++i)
        indicators[i] = i == node ? (int64_t)vars[i] : -(int64_t)vars[i];
    TididiCircuit *result = NULL;
    check(tididi_cube(vtree, indicators, 16, &result, NULL));
    return result;
}
// end: state

int main(void) {
    // begin: graph
    const size_t edges[][2] = {
        {0,1}, {1,2}, {3,2}, {4,5}, {5,6}, {6,7}, {8,9}, {9,10}, {10,11},
        {0,4}, {1,5}, {2,6}, {3,7}, {4,8}, {5,9}, {6,10}, {7,11},
        {4,0}, {5,1}, {10,6}, {11,7}, {12,13}, {13,15}, {15,14}, {14,12}
    };
    TididiVtree *vtree = NULL;
    check(tididi_vtree_balanced(32, &vtree));
    uint32_t current_vars[16], next_vars[16];
    for (uint32_t i = 0; i < 16; ++i) {
        current_vars[i] = i + 1;
        next_vars[i] = i + 17;
    }
    // end: graph

    // begin: transition
    TididiCircuit *terms[25] = {NULL}, *transition = NULL;
    for (size_t i = 0; i < 25; ++i) {
        TididiCircuit *source = state(vtree, current_vars, edges[i][0]);
        TididiCircuit *target = state(vtree, next_vars, edges[i][1]);
        check(tididi_and(source, target, &terms[i], NULL));
        release(source); release(target);
    }
    check(tididi_or_many(terms, 25, &transition, NULL));
    for (size_t i = 0; i < 25; ++i) release(terms[i]);
    TididiCircuit *reached = state(vtree, current_vars, 0);
    // end: transition

    section("image");
    // begin: image
    TididiCircuit *r = copy(reached), *t = copy(transition);
    TididiCircuit *joined = NULL, *projected = NULL, *image = NULL;
    check(tididi_and(r, t, &joined, NULL));
    check(tididi_exists(joined, current_vars, 16, &projected, NULL));
    check(tididi_rename(projected, next_vars, current_vars, 16, &image, NULL));
    uint64_t states = 0;
    check(tididi_projected_model_count(image, current_vars, 16, &states, NULL));
    printf("States after one step: %" PRIu64 "\n", states);
    // end: image
    release(r); release(t); release(joined); release(projected); release(image);

    section("fixed_point");
    // begin: fixed_point
    for (unsigned step = 1; ; ++step) {
        r = copy(reached); t = copy(transition);
        TididiCircuit *next = NULL, *renamed = NULL, *candidate = NULL, *smaller = NULL;
        check(tididi_and_exists(r, t, current_vars, 16, &next, NULL));
        check(tididi_rename(next, next_vars, current_vars, 16, &renamed, NULL));
        TididiCircuit *old_copy = copy(reached);
        check(tididi_or(old_copy, renamed, &candidate, NULL));
        check(tididi_minimize(candidate, &smaller, NULL));
        bool done = false;
        check(tididi_equivalent(reached, smaller, &done, NULL));
        size_t nodes = 0, pairs = 0;
        check(tididi_size(smaller, &nodes, &pairs));
        check(tididi_projected_model_count(smaller, current_vars, 16, &states, NULL));
        printf("Step %u: %" PRIu64 " states, %zu nodes, %zu pairs%s\n",
               step, states, nodes, pairs, done ? " (fixed point)" : "");
        release(reached); reached = smaller;
        release(r); release(t); release(next); release(renamed);
        release(old_copy); release(candidate);
        if (done) break;
    }
    // end: fixed_point

    section("unreachable");
    // begin: unreachable
    for (size_t node = 0; node < 16; ++node) {
        TididiCircuit *at_node = state(vtree, current_vars, node);
        TididiCircuit *reached_copy = copy(reached), *overlap = NULL;
        check(tididi_and(reached_copy, at_node, &overlap, NULL));
        bool reachable = false;
        check(tididi_is_sat(overlap, &reachable, NULL));
        if (!reachable) printf("Unreachable: %zu\n", node);
        release(at_node); release(reached_copy); release(overlap);
    }
    // end: unreachable
    release(reached); release(transition); tididi_vtree_free(vtree);
    return 0;
}
