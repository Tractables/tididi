/* The same traces and truth tables are replayed through all three APIs. */
#include "tididi.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static size_t line_number;
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "trace line %zu: %s\n", line_number, #x); abort(); } } while (0)
static void ok(TididiError *error) {
    if (error) { fprintf(stderr, "trace line %zu: %s\n", line_number, tididi_error_message(error)); tididi_error_free(error); abort(); }
}
static TididiCircuit *copy(TididiCircuit *f) {
    TididiCircuit *out = NULL; ok(tididi_copy(f, &out)); return out;
}
static uint64_t popcount(uint64_t value) {
    uint64_t count = 0;
    while (value) { count += value & 1; value >>= 1; }
    return count;
}
static void verify(TididiCircuit *f, unsigned n, uint64_t truth) {
    uint64_t count = 0;
    ok(tididi_model_count(f, &count, NULL)); CHECK(count == popcount(truth));
    TididiCircuit *owned = copy(f);
    TididiCounter *counter = NULL; ok(tididi_counter(owned, &counter));
    for (unsigned x = 0; x < (1u << n); ++x) {
        int64_t pins[5];
        for (unsigned v = 1; v <= n; ++v) pins[v - 1] = x & (1u << (v - 1)) ? (int64_t)v : -(int64_t)v;
        ok(tididi_counter_observe(counter, pins, n));
        ok(tididi_counter_model_count(counter, &count, NULL)); CHECK(count == ((truth >> x) & 1));
    }
    ok(tididi_counter_clear_all(counter));
    ok(tididi_counter_model_count(counter, &count, NULL)); CHECK(count == popcount(truth));
    TididiCircuit *recovered = NULL; ok(tididi_counter_finish(counter, &recovered));
    ok(tididi_counter_free(counter)); ok(tididi_circuit_free(owned)); ok(tididi_circuit_free(recovered));
    TididiLiterals *support = NULL, *implied = NULL;
    ok(tididi_support(f, &support, NULL)); ok(tididi_implied_literals(f, &implied, NULL));
    size_t expected_support = 0, expected_implied = 0;
    for (unsigned v = 1; v <= n; ++v) {
        bool depends = false;
        for (unsigned x = 0; x < (1u << n); ++x)
            depends |= (((truth >> x) ^ (truth >> (x ^ (1u << (v - 1))))) & 1) != 0;
        bool found = false;
        for (size_t i = 0; i < tididi_literals_len(support); ++i) found |= tididi_literals_data(support)[i] == (int64_t)v;
        CHECK(found == depends); expected_support += depends;
        for (int sign = -1; sign <= 1; sign += 2) {
            bool forced = true;
            for (unsigned x = 0; x < (1u << n); ++x)
                if ((truth >> x) & 1) forced &= ((x & (1u << (v - 1))) != 0) == (sign > 0);
            found = false;
            for (size_t i = 0; i < tididi_literals_len(implied); ++i) found |= tididi_literals_data(implied)[i] == sign * (int64_t)v;
            CHECK(found == forced); expected_implied += forced;
        }
    }
    CHECK(tididi_literals_len(support) == expected_support);
    CHECK(tididi_literals_len(implied) == expected_implied);
    tididi_literals_free(support); tididi_literals_free(implied);
}
int main(int argc, char **argv) {
    CHECK(argc == 2);
    FILE *input = fopen(argv[1], "r"); CHECK(input != NULL);
    TididiCircuit *circuits[256]; size_t len = 0;
    TididiVtree *vtree = NULL; unsigned n = 0;
    char line[256], op[32]; int a, b, c; uint64_t truth;
    while (fgets(line, sizeof(line), input)) {
        ++line_number;
        if (line[0] == '#') continue;
        CHECK(sscanf(line, "%31s %d %d %d %" SCNx64, op, &a, &b, &c, &truth) == 5);
        if (!strcmp(op, "case")) {
            for (size_t i = 0; i < len; ++i) ok(tididi_circuit_free(circuits[i]));
            len = 0; tididi_vtree_free(vtree); vtree = NULL;
            CHECK(a >= 1 && a <= 5); n = (unsigned)a;
            uint32_t order[5] = {1, 2, 3, 4, 5};
            if (b == 0) ok(tididi_vtree_balanced(n, &vtree));
            else ok(tididi_vtree_linear(order, n, &vtree));
            continue;
        }
        TididiCircuit *f = NULL, *left = NULL, *right = NULL, *third = NULL;
        if (!strcmp(op, "literal")) ok(tididi_literal(vtree, a, &f, NULL));
        else if (!strcmp(op, "one")) ok(tididi_one(vtree, &f));
        else if (!strcmp(op, "zero")) ok(tididi_zero(vtree, &f));
        else {
            CHECK(a >= 0 && (size_t)a < len); left = copy(circuits[a]);
            if (!strcmp(op, "and") || !strcmp(op, "or") || !strcmp(op, "xor") || !strcmp(op, "ite")) {
                CHECK(b >= 0 && (size_t)b < len); right = copy(circuits[b]);
                if (!strcmp(op, "and")) ok(tididi_and(left, right, &f, NULL));
                else if (!strcmp(op, "or")) ok(tididi_or(left, right, &f, NULL));
                else if (!strcmp(op, "xor")) ok(tididi_xor(left, right, &f, NULL));
                else { CHECK(c >= 0 && (size_t)c < len); third = copy(circuits[c]); ok(tididi_ite(left, right, third, &f, NULL)); }
            } else if (!strcmp(op, "negate")) ok(tididi_negate(left, &f, NULL));
            else if (!strcmp(op, "condition")) { int64_t literal = b; ok(tididi_condition(left, &literal, 1, &f, NULL)); }
            else if (!strcmp(op, "exists")) { uint32_t var = (uint32_t)b; ok(tididi_exists(left, &var, 1, &f, NULL)); }
            else if (!strcmp(op, "swap") || !strcmp(op, "rename")) {
                uint32_t from[] = {(uint32_t)b, (uint32_t)c}, to[] = {(uint32_t)c, (uint32_t)b};
                ok(tididi_rename(left, from, to, !strcmp(op, "swap") ? 2 : 1, &f, NULL));
            } else if (!strcmp(op, "minimize")) ok(tididi_minimize(left, &f, NULL));
            else if (!strcmp(op, "roundtrip")) {
                TididiBytes *bytes = NULL; ok(tididi_to_bytes(left, &bytes));
                ok(tididi_from_bytes(vtree, tididi_bytes_data(bytes), tididi_bytes_len(bytes), &f)); tididi_bytes_free(bytes);
            } else CHECK(false);
        }
        ok(tididi_circuit_free(left)); ok(tididi_circuit_free(right)); ok(tididi_circuit_free(third));
        verify(f, n, truth); CHECK(len < 256); circuits[len++] = f;
    }
    CHECK(!ferror(input)); CHECK(line_number > 100); fclose(input);
    for (size_t i = 0; i < len; ++i) ok(tididi_circuit_free(circuits[i]));
    tididi_vtree_free(vtree);
    return 0;
}
