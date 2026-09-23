/* Ownership and refusal sequences shared with the Rust and Python consumers. */
#include "tididi.h"
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

static size_t line_number;
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "session line %zu: %s\n", line_number, #x); abort(); } } while (0)
static void ok(TididiError *error) {
    if (error) { fprintf(stderr, "session line %zu: %s\n", line_number, tididi_error_message(error)); tididi_error_free(error); abort(); }
}
static void refused(TididiError *error, TididiErrorCode code) {
    CHECK(error != NULL); CHECK(tididi_error_code(error) == code); tididi_error_free(error);
}
static TididiCircuit *copy(TididiCircuit *f) {
    TididiCircuit *out = NULL; ok(tididi_copy(f, &out)); return out;
}
static void consumed(TididiCircuit *f) {
    bool empty = false; ok(tididi_is_consumed(f, &empty)); CHECK(empty);
    uint64_t count = 999;
    refused(tididi_model_count(f, &count, NULL), TIDIDI_ERROR_CODE_CONSUMED_CIRCUIT);
    CHECK(count == 999);
}
static TididiError *read_query(TididiCounter *counter, TididiEvaluator *evaluator, uint64_t *out, const TididiLimits *limits) {
    if (counter) return tididi_counter_model_count(counter, out, limits);
    char *text = NULL; TididiError *error = tididi_evaluator_value(evaluator, &text, limits);
    if (error) { CHECK(text == NULL); return error; }
    char *end = NULL; *out = strtoull(text, &end, 10); CHECK(*end == '\0'); tididi_string_free(text); return NULL;
}
static TididiError *observe(TididiCounter *counter, TididiEvaluator *evaluator, const int64_t *pins, size_t len) {
    return counter ? tididi_counter_observe(counter, pins, len) : tididi_evaluator_observe(evaluator, pins, len);
}
int main(int argc, char **argv) {
    CHECK(argc == 2); FILE *input = fopen(argv[1], "r"); CHECK(input != NULL);
    TididiVtree *vtree = NULL;
    TididiCircuit *f = NULL, *saved = NULL;
    TididiCounter *counter = NULL;
    TididiEvaluator *evaluator = NULL;
    TididiLimits stop = tididi_limits_default(); stop.timeout_seconds = 0;
    unsigned n = 0; bool weighted = false; uint64_t original_count = 0;
    char line[256], op[32]; int a, b, c; uint64_t expected;
    while (fgets(line, sizeof(line), input)) {
        ++line_number;
        if (line[0] == '#') continue;
        CHECK(sscanf(line, "%31s %d %d %d %" SCNx64, op, &a, &b, &c, &expected) == 5);
        int64_t pins[2] = {a, b}; size_t pin_count = b ? 2 : 1;
        if (!strcmp(op, "case")) {
            CHECK(counter == NULL && evaluator == NULL);
            ok(tididi_circuit_free(f)); ok(tididi_circuit_free(saved)); tididi_vtree_free(vtree);
            f = saved = NULL; vtree = NULL; n = (unsigned)a; weighted = c != 0;
            CHECK(n >= 2 && n <= 5);
            uint32_t order[5] = {1, 2, 3, 4, 5};
            if (b) ok(tididi_vtree_linear(order, n, &vtree)); else ok(tididi_vtree_balanced(n, &vtree));
            int64_t literals[2] = {1, 2}; ok(tididi_clause(vtree, literals, 2, &f, NULL));
            original_count = 0;
            for (uint64_t x = expected; x; x >>= 1) original_count += x & 1;
            continue;
        }
        if (!strcmp(op, "save")) saved = copy(f);
        else if (!strcmp(op, "open")) {
            if (weighted) {
                TididiWeight weights[5];
                for (unsigned v = 0; v < n; ++v) weights[v] = (TididiWeight){v + 1, "1", "1"};
                ok(tididi_evaluator(f, weights, n, &evaluator));
            } else ok(tididi_counter(f, &counter));
            consumed(f); ok(tididi_circuit_free(f)); f = NULL;
        } else if (!strcmp(op, "observe")) ok(observe(counter, evaluator, pins, pin_count));
        else if (!strcmp(op, "reject_observe")) refused(observe(counter, evaluator, pins, pin_count), TIDIDI_ERROR_CODE_INVALID_ARGUMENT);
        else if (!strcmp(op, "refuse_read") || !strcmp(op, "refuse_dirty")) {
            if (!strcmp(op, "refuse_dirty")) ok(observe(counter, evaluator, pins, pin_count));
            uint64_t out = 999;
            refused(read_query(counter, evaluator, &out, &stop), TIDIDI_ERROR_CODE_RESOURCE_LIMIT);
            CHECK(out == 999);
        } else if (!strcmp(op, "clear_one")) {
            ok(counter ? tididi_counter_clear(counter, (uint32_t)a) : tididi_evaluator_clear(evaluator, (uint32_t)a));
        } else if (!strcmp(op, "clear")) {
            ok(counter ? tididi_counter_clear_all(counter) : tididi_evaluator_clear_all(evaluator));
        } else if (!strcmp(op, "finish")) {
            ok(counter ? tididi_counter_finish(counter, &f) : tididi_evaluator_finish(evaluator, &f));
            uint64_t out = 999;
            refused(read_query(counter, evaluator, &out, NULL), TIDIDI_ERROR_CODE_INVALID_ARGUMENT);
            CHECK(out == 999);
            TididiCircuit *again = NULL;
            refused(counter ? tididi_counter_finish(counter, &again) : tididi_evaluator_finish(evaluator, &again), TIDIDI_ERROR_CODE_INVALID_ARGUMENT);
            CHECK(again == NULL);
            ok(tididi_counter_free(counter)); ok(tididi_evaluator_free(evaluator)); counter = NULL; evaluator = NULL;
        } else if (!strcmp(op, "refuse_transform")) {
            TididiCircuit *other = copy(saved), *out = NULL;
            refused(tididi_and(f, other, &out, &stop), TIDIDI_ERROR_CODE_RESOURCE_LIMIT);
            CHECK(out == NULL); consumed(f); consumed(other);
            ok(tididi_circuit_free(f)); ok(tididi_circuit_free(other)); f = NULL;
        } else if (!strcmp(op, "recover")) f = copy(saved);
        else if (!strcmp(op, "roundtrip")) {
            TididiBytes *bytes = NULL; ok(tididi_to_bytes(f, &bytes));
            ok(tididi_circuit_free(f)); f = NULL;
            ok(tididi_from_bytes(vtree, tididi_bytes_data(bytes), tididi_bytes_len(bytes), &f)); tididi_bytes_free(bytes);
        } else if (!strcmp(op, "minimize")) {
            TididiCircuit *out = NULL; ok(tididi_minimize(f, &out, NULL)); consumed(f); ok(tididi_circuit_free(f)); f = out;
        } else CHECK(false);
        uint64_t answer = 0;
        if (counter || evaluator) ok(read_query(counter, evaluator, &answer, NULL));
        else ok(tididi_model_count(f ? f : saved, &answer, NULL));
        CHECK(answer == expected);
        if (saved) { ok(tididi_model_count(saved, &answer, NULL)); CHECK(answer == original_count); }
    }
    CHECK(!ferror(input)); CHECK(line_number > 100); fclose(input);
    CHECK(counter == NULL && evaluator == NULL);
    ok(tididi_circuit_free(f)); ok(tididi_circuit_free(saved)); tididi_vtree_free(vtree);
    return 0;
}
