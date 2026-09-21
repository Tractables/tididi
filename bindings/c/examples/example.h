// scenario: docs/scenarios.md#first-circuit
#ifndef TIDIDI_EXAMPLE_H
#define TIDIDI_EXAMPLE_H
#include "tididi.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

/* Shared housekeeping for these standalone programs. */
static inline void check(TididiError *error) {
    if (error) {
        fprintf(stderr, "%s\n", tididi_error_message(error));
        tididi_error_free(error);
        exit(EXIT_FAILURE);
    }
}
static inline TididiCircuit *copy(const TididiCircuit *value) {
    TididiCircuit *result = NULL;
    check(tididi_copy(value, &result));
    return result;
}
static inline void release(TididiCircuit *value) { check(tididi_circuit_free(value)); }
static inline uint64_t count(const TididiCircuit *value) {
    uint64_t result = 0;
    check(tididi_model_count(value, &result, NULL));
    return result;
}
/* The documentation captures these sections from actual program output. */
static inline void section(const char *name) { printf("=== %s ===\n", name); }
#endif
