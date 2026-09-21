/* tididi C interface
 *
 * TididiError* results: NULL is success. Otherwise read the error, then free it.
 * Owned output pointers must be initialized to NULL; scalar outputs need live storage.
 * All output storage must be separate from input storage.
 * Transformations consume circuit payloads. Free their empty handles afterwards.
 * Queries borrow circuits. Use tididi_copy before a transformation to retain an input.
 * Invalid arguments preserve inputs; execution failures may leave inputs consumed.
 * Arrays may be NULL only at length zero. Strings are NUL-terminated UTF-8.
 * Handles must be live objects from this library; synchronize access across threads.
 * Returned allocations require their matching tididi_*_free function, never free().
 * Callbacks and userdata are borrowed for the call; callbacks must not unwind or longjmp.
 * Full guide: https://tractables.github.io/tididi/c/
 */


#ifndef TIDIDI_H
#define TIDIDI_H

/* Generated from the Rust C binding. Regenerate with python bindings/c/check.py --write-header. */

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * Classification of an operation failure.
 */
typedef enum TididiErrorCode {
  /**
   * Invalid input, incompatible domains, or a finished counter.
   */
  TIDIDI_ERROR_CODE_INVALID_ARGUMENT = 1,
  /**
   * A live circuit handle no longer contains a circuit.
   */
  TIDIDI_ERROR_CODE_CONSUMED_CIRCUIT = 2,
  /**
   * A callback tried to mutate or free an actively borrowed handle.
   */
  TIDIDI_ERROR_CODE_BORROW_CONFLICT = 3,
  /**
   * The operation exceeded its charged-storage budget.
   */
  TIDIDI_ERROR_CODE_MEMORY_LIMIT = 4,
  /**
   * The operation reached its timeout or output-node cap.
   */
  TIDIDI_ERROR_CODE_RESOURCE_LIMIT = 5,
  /**
   * A count or index cannot fit the requested integer representation.
   */
  TIDIDI_ERROR_CODE_OVERFLOW = 6,
  /**
   * An unexpected Rust panic was caught at the boundary.
   */
  TIDIDI_ERROR_CODE_INTERNAL_PANIC = 7,
} TididiErrorCode;

/**
 * An owned byte buffer. Read its data/length and free it with tididi_bytes_free.
 */
typedef struct TididiBytes TididiBytes;

/**
 * An owned circuit handle. Transformations consume its payload, not this handle.
 * Free every handle, including consumed ones, with tididi_circuit_free.
 */
typedef struct TididiCircuit TididiCircuit;

/**
 * A reusable evidence counter that owns its circuit until finish. Free its handle even after finish.
 */
typedef struct TididiCounter TididiCounter;

/**
 * An owned error. Read its code/message, then call tididi_error_free.
 */
typedef struct TididiError TididiError;

/**
 * An owned list of signed, one-based literals. Its data remains valid until the list is freed.
 */
typedef struct TididiLiterals TididiLiterals;

/**
 * An owned snapshot of internal-node sizes, independent of the circuit's lifetime.
 */
typedef struct TididiNodeSizes TididiNodeSizes;

/**
 * A shared vtree handle. Circuits retain the vtree after this handle is freed.
 */
typedef struct TididiVtree TididiVtree;

/**
 * Per-operation limits. Pass NULL for unlimited work or initialize with tididi_limits_default.
 */
typedef struct TididiLimits {
  /**
   * Charged operation storage in bytes; UINT64_MAX means unlimited.
   */
  uint64_t memory_bytes;
  /**
   * Maximum emitted circuit nodes; UINT64_MAX means unlimited.
   */
  uint64_t output_nodes;
  /**
   * Cooperative timeout in seconds; -1 means unlimited. Other values must be finite and nonnegative.
   */
  double timeout_seconds;
} TididiLimits;

/**
 * One partial assignment used in an update. Omitted variables are free; an empty cube matches all assignments.
 */
typedef struct TididiCube {
  const int64_t *literals;
  size_t len;
} TididiCube;

/**
 * One internal node's storage size. IDs are local and may change after transformations.
 */
typedef struct TididiNodeSize {
  uint32_t vtree_node;
  size_t local_node;
  size_t pairs;
} TididiNodeSize;

/**
 * Exact negative/positive literal weights, written as integers or fractions such as "3/5".
 */
typedef struct TididiWeight {
  uint32_t variable;
  const char *negative;
  const char *positive;
} TididiWeight;

/**
 * Numeric evaluation callbacks. Every callback is required; userdata is borrowed for the call.
 * leaf sign is 1 (true), 0 (false), or -1 (free). Callbacks must not unwind or longjmp.
 * add combines disjoint alternatives; mul combines independent variable groups.
 * Both operations must be associative and commutative, mul must distribute over add,
 * and zero must be the identity for add and absorbing for mul.
 * leaf(v, -1) must equal add(leaf(v, 0), leaf(v, 1)): a free variable includes both signs.
 * Equivalent circuits need not evaluate equally if these laws are violated.
 * Double arithmetic approximates these laws; use weighted_count for exact rational sums.
 */
typedef struct TididiAlgebra {
  void *userdata;
  double (*zero)(void*);
  double (*leaf)(void*, uint32_t, int8_t);
  double (*add)(void*, double, double);
  double (*mul)(void*, double, double);
} TididiAlgebra;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * Return an error's code. error must be nonnull and live.
 */
enum TididiErrorCode tididi_error_code(const struct TididiError *error);

/**
 * Borrow a UTF-8 message until error is freed. error must be nonnull and live.
 */
const char *tididi_error_message(const struct TididiError *error);

/**
 * Free an error; NULL is accepted.
 */
void tididi_error_free(struct TididiError *error);

/**
 * Free an unchanged string returned by this library; NULL is accepted. Never use free().
 */
void tididi_string_free(char *value);

/**
 * Return the binding version as a borrowed static string. Do not free it.
 */
const char *tididi_version(void);

/**
 * Return unlimited limits. Modify fields before passing their address to an operation.
 */
struct TididiLimits tididi_limits_default(void);

/**
 * Build a balanced vtree over variables 1 through n; n must be positive. out must point to NULL.
 */
struct TididiError *tididi_vtree_balanced(uint32_t n, struct TididiVtree **out);

/**
 * Build a balanced vtree over distinct positive variable IDs in the given leaf order.
 */
struct TididiError *tididi_vtree_balanced_over(const uint32_t *order,
                                               size_t len,
                                               struct TididiVtree **out);

/**
 * Build a right-linear vtree over distinct positive IDs in the supplied order.
 */
struct TididiError *tididi_vtree_linear(const uint32_t *order,
                                        size_t len,
                                        struct TididiVtree **out);

/**
 * Join disjoint vtrees under a new root. Borrows both inputs; the result is a new domain.
 */
struct TididiError *tididi_vtree_join(const struct TididiVtree *left,
                                      const struct TididiVtree *right,
                                      struct TididiVtree **out);

/**
 * Serialize the vtree as an owned UTF-8 string. Free it with tididi_string_free.
 */
struct TididiError *tididi_vtree_to_text(const struct TididiVtree *value, char **out);

/**
 * Parse vtree text. Load related circuits onto this one returned domain.
 */
struct TididiError *tididi_vtree_from_text(const char *value, struct TididiVtree **out);

/**
 * Release idle operation buffers without changing circuits.
 */
struct TididiError *tididi_vtree_clear_scratch(const struct TididiVtree *value);

/**
 * Free this vtree handle; NULL is accepted. Existing circuits retain their shared vtree.
 */
void tididi_vtree_free(struct TididiVtree *value);

/**
 * Return another handle to a live circuit's shared vtree. Free the returned handle normally.
 */
struct TididiError *tididi_circuit_vtree(const struct TididiCircuit *value,
                                         struct TididiVtree **out);

/**
 * Copy diagram storage while sharing the vtree. The input remains usable.
 */
struct TididiError *tididi_copy(const struct TididiCircuit *value, struct TididiCircuit **out);

/**
 * Report whether an operation consumed this handle's payload. The handle itself must still be live.
 */
struct TididiError *tididi_is_consumed(const struct TididiCircuit *value, bool *out);

/**
 * Free a circuit handle, including a consumed one. NULL is accepted.
 * Returns BorrowConflict without freeing if a callback attempts to free an active handle.
 */
struct TididiError *tididi_circuit_free(struct TididiCircuit *value);

/**
 * Construct a signed, one-based literal; zero is invalid. out must point to NULL.
 */
struct TididiError *tididi_literal(const struct TididiVtree *tree,
                                   int64_t value,
                                   struct TididiCircuit **out,
                                   const struct TididiLimits *config);

/**
 * Construct the constant true function over all vtree variables.
 */
struct TididiError *tididi_one(const struct TididiVtree *tree, struct TididiCircuit **out);

/**
 * Construct the constant false function over all vtree variables.
 */
struct TididiError *tididi_zero(const struct TididiVtree *tree, struct TididiCircuit **out);

/**
 * Conjoin signed literals, each variable occurring once. An empty cube is true; omitted variables are free.
 */
struct TididiError *tididi_cube(const struct TididiVtree *tree,
                                const int64_t *values,
                                size_t len,
                                struct TididiCircuit **out,
                                const struct TididiLimits *config);

/**
 * Disjoin signed literals. An empty clause is false.
 */
struct TididiError *tididi_clause(const struct TididiVtree *tree,
                                  const int64_t *values,
                                  size_t len,
                                  struct TididiCircuit **out,
                                  const struct TididiLimits *config);

/**
 * Conjoin two circuits, consuming both. Preflight errors preserve inputs; execution failures consume them.
 */
struct TididiError *tididi_and(struct TididiCircuit *left,
                               struct TididiCircuit *right,
                               struct TididiCircuit **out,
                               const struct TididiLimits *config);

/**
 * Disjoin two circuits, consuming both. Operands must be distinct handles sharing one vtree.
 */
struct TididiError *tididi_or(struct TididiCircuit *left,
                              struct TididiCircuit *right,
                              struct TididiCircuit **out,
                              const struct TididiLimits *config);

/**
 * Exclusive-or two circuits, consuming both.
 */
struct TididiError *tididi_xor(struct TididiCircuit *left,
                               struct TididiCircuit *right,
                               struct TididiCircuit **out,
                               const struct TididiLimits *config);

/**
 * Complement a circuit, consuming it.
 */
struct TididiError *tididi_negate(struct TididiCircuit *value,
                                  struct TididiCircuit **out,
                                  const struct TididiLimits *config);

/**
 * Minimize without changing the function. Consumes the old payload and returns a canonical circuit for its vtree.
 */
struct TididiError *tididi_minimize(struct TididiCircuit *value,
                                    struct TididiCircuit **out,
                                    const struct TididiLimits *config);

/**
 * Substitute literal values, consuming the circuit. Repeats are ignored; opposite signs produce false.
 * Substituted variables remain free in the counting universe; use a counter to count under observations.
 */
struct TididiError *tididi_condition(struct TididiCircuit *value,
                                     const int64_t *assignments,
                                     size_t len,
                                     struct TididiCircuit **out,
                                     const struct TididiLimits *config);

/**
 * Existentially quantify variables, consuming the circuit. They remain free in the vtree's counting universe.
 */
struct TididiError *tididi_exists(struct TididiCircuit *value,
                                  const uint32_t *vars,
                                  size_t len,
                                  struct TididiCircuit **out,
                                  const struct TididiLimits *config);

/**
 * Conjoin then quantify in one call, consuming both inputs. Equivalent to and followed by exists.
 */
struct TididiError *tididi_and_exists(struct TididiCircuit *left,
                                      struct TididiCircuit *right,
                                      const uint32_t *vars,
                                      size_t len,
                                      struct TididiCircuit **out,
                                      const struct TididiLimits *config);

/**
 * Rename from[i] to to[i] simultaneously, consuming the circuit. Swaps and cycles are simultaneous.
 * Each source occurs once; several sources may share a target. The vtree stays unchanged.
 */
struct TididiError *tididi_rename(struct TididiCircuit *value,
                                  const uint32_t *from,
                                  const uint32_t *to,
                                  size_t len,
                                  struct TididiCircuit **out,
                                  const struct TididiLimits *config);

/**
 * Union a nonempty array of distinct circuit handles, consuming every payload.
 */
struct TididiError *tididi_or_many(struct TididiCircuit *const *values,
                                   size_t len,
                                   struct TididiCircuit **out,
                                   const struct TididiLimits *config);

/**
 * Build if-then-else, consuming three distinct circuit handles sharing one vtree.
 */
struct TididiError *tididi_ite(struct TididiCircuit *condition,
                               struct TididiCircuit *yes,
                               struct TididiCircuit *no,
                               struct TididiCircuit **out,
                               const struct TididiLimits *config);

/**
 * Build a set of row-major Boolean rows. Every cell is 0 or 1; duplicates count once.
 * rows contains nrows*nvars bytes. Other vtree variables are free.
 */
struct TididiError *tididi_from_models(const struct TididiVtree *tree,
                                       const uint32_t *vars,
                                       size_t nvars,
                                       const uint8_t *rows,
                                       size_t nrows,
                                       struct TididiCircuit **out,
                                       const struct TididiLimits *config);

/**
 * Insert cubes, then remove cubes; consume the old circuit and return its minimized replacement.
 * Contradictory cubes change nothing. Limits apply to individual updates and final minimization.
 */
struct TididiError *tididi_update(struct TididiCircuit *value,
                                  const struct TididiCube *insert,
                                  size_t ninsert,
                                  const struct TididiCube *remove,
                                  size_t nremove,
                                  struct TididiCircuit **out,
                                  const struct TididiLimits *config);

/**
 * Count models over every vtree variable, borrowing the circuit. Overflow leaves out unchanged.
 */
struct TididiError *tididi_model_count(const struct TididiCircuit *value,
                                       uint64_t *out,
                                       const struct TididiLimits *config);

/**
 * Count exactly, returning an owned decimal string of arbitrary length. Free it with tididi_string_free.
 */
struct TididiError *tididi_model_count_decimal(const struct TididiCircuit *value,
                                               char **out,
                                               const struct TididiLimits *config);

/**
 * Count distinct assignments to the selected variables that extend to a model. Borrows the circuit.
 */
struct TididiError *tididi_projected_model_count(const struct TididiCircuit *value,
                                                 const uint32_t *vars,
                                                 size_t len,
                                                 uint64_t *out,
                                                 const struct TididiLimits *config);

/**
 * Count distinct projections exactly, returning an owned decimal string.
 */
struct TididiError *tididi_projected_model_count_decimal(const struct TididiCircuit *value,
                                                         const uint32_t *vars,
                                                         size_t len,
                                                         char **out,
                                                         const struct TididiLimits *config);

/**
 * Whether at least one assignment satisfies the circuit. Borrows it.
 */
struct TididiError *tididi_is_sat(const struct TididiCircuit *value,
                                  bool *out,
                                  const struct TididiLimits *config);

/**
 * Compare Boolean functions without consuming either handle. Aliases are allowed; vtrees must match.
 */
struct TididiError *tididi_equivalent(const struct TididiCircuit *left,
                                      const struct TididiCircuit *right,
                                      bool *out,
                                      const struct TididiLimits *config);

/**
 * Whether every model of left satisfies right. Borrows both handles; vtrees must match.
 */
struct TididiError *tididi_implies(const struct TididiCircuit *left,
                                   const struct TididiCircuit *right,
                                   bool *out,
                                   const struct TididiLimits *config);

/**
 * Return the size of a live, nonnull literal list.
 */
size_t tididi_literals_len(const struct TididiLiterals *value);

/**
 * Borrow the signed literal array. Do not write to or free it separately.
 */
const int64_t *tididi_literals_data(const struct TididiLiterals *value);

/**
 * Free a literal list; NULL is accepted.
 */
void tididi_literals_free(struct TididiLiterals *value);

/**
 * Return literals implied by the function. False implies both signs of every vtree variable.
 */
struct TididiError *tididi_implied_literals(const struct TididiCircuit *value,
                                            struct TididiLiterals **out,
                                            const struct TididiLimits *config);

/**
 * Return one complete satisfying assignment, or an empty list when unsatisfiable. Borrows the circuit.
 */
struct TididiError *tididi_satisfying_assignment(const struct TididiCircuit *value,
                                                 struct TididiLiterals **out,
                                                 const struct TididiLimits *config);

/**
 * Return the function's support as positive variable IDs in a literal list. Borrows the circuit.
 */
struct TididiError *tididi_support(const struct TididiCircuit *value,
                                   struct TididiLiterals **out,
                                   const struct TididiLimits *config);

/**
 * Report stored nodes and child pairs, not models. Implicit leaves are excluded from the node count.
 */
struct TididiError *tididi_size(const struct TididiCircuit *value,
                                size_t *nodes,
                                size_t *pairs);

/**
 * Snapshot the size of every stored internal node, borrowing the circuit.
 */
struct TididiError *tididi_node_sizes(const struct TididiCircuit *value,
                                      struct TididiNodeSizes **out);

/**
 * Return the length of a live, nonnull node-size snapshot.
 */
size_t tididi_node_sizes_len(const struct TididiNodeSizes *value);

/**
 * Borrow the snapshot's rows until it is freed. Do not modify or free the array separately.
 */
const struct TididiNodeSize *tididi_node_sizes_data(const struct TididiNodeSizes *value);

/**
 * Free a snapshot; NULL is accepted.
 */
void tididi_node_sizes_free(struct TididiNodeSizes *value);

/**
 * Move a circuit into an evidence counter. Copy first if the original must remain usable.
 */
struct TididiError *tididi_counter(struct TididiCircuit *value, struct TididiCounter **out);

/**
 * Observe signed literals. Later observations replace pins for named variables and retain the other pins.
 */
struct TididiError *tididi_counter_observe(struct TididiCounter *value,
                                           const int64_t *values,
                                           size_t len);

/**
 * Remove the observation for one variable.
 */
struct TididiError *tididi_counter_clear(struct TididiCounter *value, uint32_t var);

/**
 * Remove all observations, retaining the circuit and reusable counter.
 */
struct TididiError *tididi_counter_clear_all(struct TididiCounter *value);

/**
 * Count assignments consistent with observations. Does not consume the counter.
 */
struct TididiError *tididi_counter_model_count(struct TididiCounter *value,
                                               uint64_t *out,
                                               const struct TididiLimits *config);

/**
 * Count under observations exactly, returning an owned decimal string.
 */
struct TididiError *tididi_counter_model_count_decimal(struct TididiCounter *value,
                                                       char **out,
                                                       const struct TididiLimits *config);

/**
 * Discard observations/cache and return the original circuit. Closes the counter; its handle still needs freeing.
 */
struct TididiError *tididi_counter_finish(struct TididiCounter *value,
                                          struct TididiCircuit **out);

/**
 * Free a counter handle, whether open or finished. NULL is accepted; active handles return BorrowConflict.
 */
struct TididiError *tididi_counter_free(struct TididiCounter *value);

/**
 * Exact weighted sum, returned as an owned integer/fraction string. Supply weights for every vtree variable.
 * Borrows the circuit. The string is freed with tididi_string_free.
 */
struct TididiError *tididi_weighted_count(const struct TididiCircuit *value,
                                          const struct TididiWeight *values,
                                          size_t len,
                                          char **out,
                                          const struct TididiLimits *config);

/**
 * Divide two exact weighted sums, borrowing both circuits on one shared vtree.
 * For P(query|evidence), numerator must already represent query AND evidence.
 * A zero denominator returns InvalidArgument; out is an owned integer/fraction string.
 */
struct TididiError *tididi_weighted_ratio(const struct TididiCircuit *numerator,
                                          const struct TididiCircuit *denominator,
                                          const struct TididiWeight *values,
                                          size_t len,
                                          char **out,
                                          const struct TididiLimits *config);

/**
 * Evaluate a numeric algebra while borrowing the circuit. Callback values use double precision.
 * Reentrant attempts to consume or free this circuit return BorrowConflict.
 */
struct TididiError *tididi_evaluate_f64(const struct TididiCircuit *value,
                                        const struct TididiAlgebra *algebra,
                                        double *out,
                                        const struct TididiLimits *config);

/**
 * Return the size of a live, nonnull buffer.
 */
size_t tididi_bytes_len(const struct TididiBytes *value);

/**
 * Borrow a buffer's data until it is freed. Never modify it or free it separately.
 */
const uint8_t *tididi_bytes_data(const struct TididiBytes *value);

/**
 * Free an owned byte buffer; NULL is accepted.
 */
void tididi_bytes_free(struct TididiBytes *value);

/**
 * Serialize a circuit without consuming it. Save its vtree separately.
 */
struct TididiError *tididi_to_bytes(const struct TididiCircuit *value, struct TididiBytes **out);

/**
 * Load circuit bytes onto an existing vtree. Use one shared vtree for related circuits.
 */
struct TididiError *tididi_from_bytes(const struct TididiVtree *tree,
                                      const uint8_t *data,
                                      size_t len,
                                      struct TididiCircuit **out);

/**
 * Export a circuit as owned Graphviz text. Does not consume it.
 */
struct TididiError *tididi_to_dot(const struct TididiCircuit *value, char **out);

/**
 * Export a vtree as owned Graphviz text.
 */
struct TididiError *tididi_vtree_to_dot(const struct TididiVtree *value, char **out);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  /* TIDIDI_H */
