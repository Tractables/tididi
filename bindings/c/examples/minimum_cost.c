// scenario: docs/scenarios.md#minimum-cost
#include "example.h"
#include <math.h>

// begin: algebra
static double impossible(void *data) { (void)data; return INFINITY; }
static double literal_cost(void *data, uint32_t variable, int8_t sign) {
    const double *prices = (const double *)data;
    if (sign == 1) return prices[variable - 1];
    if (sign == 0) return 0;
    return fmin(0, prices[variable - 1]); /* A free variable chooses its cheaper sign. */
}
static double cheaper(void *data, double a, double b) { (void)data; return fmin(a, b); }
static double combined(void *data, double a, double b) { (void)data; return a + b; }
// end: algebra

int main(void) {
    TididiVtree *vtree = NULL;
    check(tididi_vtree_balanced(4, &vtree));
    TididiCircuit *a = NULL, *b = NULL, *rules = NULL;
    const int64_t destination[] = {1,2}, encryption[] = {-2,3};
    check(tididi_clause(vtree, destination, 2, &a, NULL));
    check(tididi_clause(vtree, encryption, 2, &b, NULL));
    check(tididi_and(a, b, &rules, NULL));
    release(a); release(b);

    section("cost");
    // begin: cost
    double prices[] = {5, 2, 1, 0};
    TididiAlgebra algebra = {prices, impossible, literal_cost, cheaper, combined};
    double cost = 0;
    check(tididi_evaluate_f64(rules, &algebra, &cost, NULL));
    printf("Minimum cost: %.0f\n", cost);
    prices[0] = 1;
    check(tididi_evaluate_f64(rules, &algebra, &cost, NULL));
    printf("With local storage discounted: %.0f\n", cost);
    // end: cost

    section("conflict");
    // begin: conflict
    const int64_t choice[] = {2,-3};
    TididiCircuit *invalid_choice = NULL, *rules_copy = copy(rules), *selected = NULL;
    check(tididi_cube(vtree, choice, 2, &invalid_choice, NULL));
    check(tididi_and(rules_copy, invalid_choice, &selected, NULL));
    check(tididi_evaluate_f64(selected, &algebra, &cost, NULL));
    printf("Remote without encryption is feasible: %s\n", isfinite(cost) ? "true" : "false");
    // end: conflict
    release(invalid_choice); release(rules_copy); release(selected); release(rules);
    tididi_vtree_free(vtree);
    return 0;
}
