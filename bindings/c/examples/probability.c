// scenario: docs/scenarios.md#probability
#include "example.h"

int main(void) {
    TididiVtree *vtree = NULL;
    check(tididi_vtree_balanced(3, &vtree));
    // begin: model
    enum { RAIN = 1, SPRINKLER, WIND };
    const int64_t causes[] = {RAIN, SPRINKLER};
    TididiCircuit *wet = NULL, *rain = NULL, *rain_and_wet = NULL;
    check(tididi_clause(vtree, causes, 2, &wet, NULL));
    check(tididi_literal(vtree, RAIN, &rain, NULL));
    TididiCircuit *wet_copy = copy(wet);
    check(tididi_and(rain, wet_copy, &rain_and_wet, NULL));
    // end: model
    release(rain); release(wet_copy);

    section("probability");
    // begin: probability
    TididiWeight weights[] = {
        {RAIN, "4/5", "1/5"}, {SPRINKLER, "9/10", "1/10"}, {WIND, "3/5", "2/5"}
    };
    char *mass = NULL, *conditional = NULL;
    check(tididi_weighted_count(wet, weights, 3, &mass, NULL));
    check(tididi_weighted_ratio(rain_and_wet, wet, weights, 3, &conditional, NULL));
    printf("P(wet) = %s\nP(rain | wet) = %s\n", mass, conditional);
    tididi_string_free(mass); tididi_string_free(conditional);
    // end: probability

    section("new_prior");
    // begin: new_prior
    weights[0].negative = "2/5";
    weights[0].positive = "3/5";
    mass = NULL; conditional = NULL;
    check(tididi_weighted_count(wet, weights, 3, &mass, NULL));
    check(tididi_weighted_ratio(rain_and_wet, wet, weights, 3, &conditional, NULL));
    printf("P(wet) = %s\nP(rain | wet) = %s\n", mass, conditional);
    tididi_string_free(mass); tididi_string_free(conditional);
    // end: new_prior
    release(wet); release(rain_and_wet); tididi_vtree_free(vtree);
    return 0;
}
