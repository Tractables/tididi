#include "tididi.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); abort(); } } while (0)
static void ok(TididiError *error) {
    if (error) { fprintf(stderr, "%s\n", tididi_error_message(error)); tididi_error_free(error); abort(); }
}
static void failure(TididiError *error, TididiErrorCode expected) {
    CHECK(error != NULL);
    if (tididi_error_code(error) != expected) fprintf(stderr, "%s\n", tididi_error_message(error));
    CHECK(tididi_error_code(error) == expected);
    tididi_error_free(error);
}
static TididiCircuit *lit(TididiVtree *v, int64_t id) {
    TididiCircuit *f = NULL; ok(tididi_literal(v, id, &f, NULL)); return f;
}
static uint64_t count(TididiCircuit *f) { uint64_t n = 0; ok(tididi_model_count(f, &n, NULL)); return n; }
static bool consumed(TididiCircuit *f) { bool b = false; ok(tididi_is_consumed(f, &b)); return b; }
static void ownership(void) {
    TididiVtree *v = NULL, *other = NULL;
    ok(tididi_vtree_balanced(3, &v)); ok(tididi_vtree_balanced(3, &other));
    TididiCircuit *a = lit(v, 1), *b = lit(v, 2), *alien = lit(other, 1), *f = NULL, *copy = NULL;
    failure(tididi_and(a, a, &f, NULL), TIDIDI_ERROR_CODE_INVALID_ARGUMENT);
    failure(tididi_and(a, alien, &f, NULL), TIDIDI_ERROR_CODE_INVALID_ARGUMENT);
    failure(tididi_and(a, b, NULL, NULL), TIDIDI_ERROR_CODE_INVALID_ARGUMENT);
    failure(tididi_and(a, b, &a, NULL), TIDIDI_ERROR_CODE_INVALID_ARGUMENT);
    CHECK(count(a) == 4 && !consumed(b));
    ok(tididi_and(a, b, &f, NULL));
    CHECK(count(f) == 2 && consumed(a) && consumed(b));
    uint64_t unchanged = 123;
    failure(tididi_model_count(a, &unchanged, NULL), TIDIDI_ERROR_CODE_CONSUMED_CIRCUIT);
    CHECK(unchanged == 123);
    failure(tididi_and(f, a, &copy, NULL), TIDIDI_ERROR_CODE_CONSUMED_CIRCUIT);
    CHECK(!consumed(f));
    ok(tididi_copy(f, &copy));
    bool same = false; ok(tididi_equivalent(f, f, &same, NULL)); CHECK(same);
    ok(tididi_implies(copy, f, &same, NULL)); CHECK(same);
    int64_t invalid_assignment[] = {1, -1, 4}; TididiCircuit *bad = NULL;
    failure(tididi_condition(f, invalid_assignment, sizeof(invalid_assignment) / sizeof(*invalid_assignment), &bad, NULL), TIDIDI_ERROR_CODE_INVALID_ARGUMENT);
    uint32_t absent = 4;
    failure(tididi_exists(f, &absent, 1, &bad, NULL), TIDIDI_ERROR_CODE_INVALID_ARGUMENT);
    CHECK(count(f) == 2);
    TididiLimits limit = tididi_limits_default(); limit.timeout_seconds = 0;
    failure(tididi_and(f, copy, &bad, &limit), TIDIDI_ERROR_CODE_RESOURCE_LIMIT);
    CHECK(consumed(f) && consumed(copy) && bad == NULL);
    TididiCircuit *handles[] = {a,b,alien,f,copy};
    for (size_t i=0; i<5; ++i) ok(tididi_circuit_free(handles[i]));
    ok(tididi_circuit_free(NULL)); tididi_vtree_free(v); tididi_vtree_free(other);
}
/* Invalid later operands must not consume the valid handles checked before them. */
static void multi_operand_preflight(void) {
    TididiVtree *vtree = NULL, *other = NULL;
    ok(tididi_vtree_balanced(3, &vtree));
    ok(tididi_vtree_balanced(3, &other));
    TididiCircuit *a = lit(vtree, 1), *b = lit(vtree, 2), *used = lit(vtree, 3);
    TididiCircuit *alien = lit(other, 3), *negative = NULL, *out = NULL;
    ok(tididi_negate(used, &negative, NULL));
    TididiCircuit *invalid_last[] = {a, used, NULL, alien};
    TididiErrorCode errors[] = {TIDIDI_ERROR_CODE_INVALID_ARGUMENT,
        TIDIDI_ERROR_CODE_CONSUMED_CIRCUIT, TIDIDI_ERROR_CODE_INVALID_ARGUMENT,
        TIDIDI_ERROR_CODE_INVALID_ARGUMENT};
    for (size_t i = 0; i < sizeof(invalid_last) / sizeof(*invalid_last); ++i) {
        TididiCircuit *inputs[] = {a, b, invalid_last[i]};
        failure(tididi_or_many(inputs, 3, &out, NULL), errors[i]);
        failure(tididi_ite(a, b, invalid_last[i], &out, NULL), errors[i]);
        CHECK(out == NULL && count(a) == 4 && count(b) == 4);
    }
    TididiCircuit *inputs[] = {a, b, negative};
    failure(tididi_or_many(inputs, 3, NULL, NULL), TIDIDI_ERROR_CODE_INVALID_ARGUMENT);
    failure(tididi_ite(a, b, negative, NULL, NULL), TIDIDI_ERROR_CODE_INVALID_ARGUMENT);
    out = alien;
    failure(tididi_or_many(inputs, 3, &out, NULL), TIDIDI_ERROR_CODE_INVALID_ARGUMENT);
    failure(tididi_ite(a, b, negative, &out, NULL), TIDIDI_ERROR_CODE_INVALID_ARGUMENT);
    CHECK(out == alien && count(alien) == 4);
    out = NULL;
    TididiLimits limits = tididi_limits_default();
    limits.timeout_seconds = -2;
    failure(tididi_or_many(inputs, 3, &out, &limits), TIDIDI_ERROR_CODE_INVALID_ARGUMENT);
    failure(tididi_ite(a, b, negative, &out, &limits), TIDIDI_ERROR_CODE_INVALID_ARGUMENT);
    CHECK(out == NULL && count(a) == 4 && count(b) == 4 && count(negative) == 4);
    ok(tididi_or_many(inputs, 3, &out, NULL));
    CHECK(count(out) == 7 && consumed(a) && consumed(b) && consumed(negative));
    TididiCircuit *handles[] = {a, b, used, alien, negative, out};
    for (size_t i = 0; i < sizeof(handles) / sizeof(*handles); ++i) ok(tididi_circuit_free(handles[i]));
    tididi_vtree_free(vtree);
    tididi_vtree_free(other);
}

static void queries(void) {
    TididiVtree *v = NULL; ok(tididi_vtree_balanced(3, &v));
    TididiCircuit *f = NULL, *impossible = NULL;
    int64_t clause[] = {1, 2}; ok(tididi_clause(v, clause, 2, &f, NULL)); ok(tididi_zero(v, &impossible));
    TididiLiterals *support = NULL, *forced = NULL, *witness = NULL;
    ok(tididi_support(f, &support, NULL)); CHECK(tididi_literals_len(support) == 2);
    CHECK(tididi_literals_data(support)[0] == 1 && tididi_literals_data(support)[1] == 2);
    ok(tididi_implied_literals(impossible, &forced, NULL)); CHECK(tididi_literals_len(forced) == 6);
    for (int64_t literal=-3; literal<=3; ++literal) {
        if (!literal) continue;
        size_t found=0; for (size_t i=0;i<6;++i) found += tididi_literals_data(forced)[i] == literal;
        CHECK(found == 1);
    }
    ok(tididi_satisfying_assignment(f, &witness, NULL)); CHECK(tididi_literals_len(witness) == 3);
    TididiCircuit *selected = NULL;
    ok(tididi_cube(v, tididi_literals_data(witness), tididi_literals_len(witness), &selected, NULL));
    bool yes = false; ok(tididi_implies(selected, f, &yes, NULL)); CHECK(yes);
    uint32_t vars[] = {1,2}; uint64_t n=0;
    ok(tididi_projected_model_count(f, vars, 2, &n, NULL)); CHECK(n==3);
    TididiCircuit *projected = NULL; ok(tididi_exists(f, vars, 2, &projected, NULL)); CHECK(count(projected)==8);
    tididi_literals_free(support); tididi_literals_free(forced); tididi_literals_free(witness);
    ok(tididi_circuit_free(f)); ok(tididi_circuit_free(impossible)); ok(tididi_circuit_free(selected)); ok(tididi_circuit_free(projected)); tididi_vtree_free(v);
}
static void big_counts_and_rows(void) {
    TididiVtree *v = NULL; TididiCircuit *f = NULL;
    ok(tididi_vtree_balanced(257, &v)); ok(tididi_one(v, &f));
    uint64_t n=17; failure(tididi_model_count(f, &n, NULL), TIDIDI_ERROR_CODE_OVERFLOW); CHECK(n==17);
    char *decimal = NULL; ok(tididi_model_count_decimal(f, &decimal, NULL));
    CHECK(strcmp(decimal,"231584178474632390847141970017375815706539969331281128078915168015826259279872")==0);
    tididi_string_free(decimal); ok(tididi_circuit_free(f)); tididi_vtree_free(v);
    v=NULL; f=NULL; ok(tididi_vtree_balanced(65,&v));
    uint32_t vars[65]; uint8_t rows[3*65]={0};
    for(uint32_t i=0;i<65;++i) vars[i]=i+1;
    rows[65+64]=1; rows[2*65+64]=1;
    ok(tididi_from_models(v,vars,65,rows,3,&f,NULL)); CHECK(count(f)==2);
    TididiCircuit *updated=NULL; int64_t removed[]={65}; TididiCube cube={removed,1};
    ok(tididi_update(f,NULL,0,&cube,1,&updated,NULL)); CHECK(count(updated)==1);
    TididiCircuit *bad=NULL; rows[0]=2;
    failure(tididi_from_models(v,vars,65,rows,3,&bad,NULL),TIDIDI_ERROR_CODE_INVALID_ARGUMENT);
    ok(tididi_circuit_free(f));ok(tididi_circuit_free(updated));tididi_vtree_free(v);
}
static void counters_and_persistence(void) {
    TididiVtree *v=NULL;ok(tididi_vtree_balanced(3,&v));
    TididiCircuit *f=NULL;int64_t disjunction[]={1,2};ok(tididi_clause(v,disjunction,2,&f,NULL));
    TididiBytes *bytes=NULL;ok(tididi_to_bytes(f,&bytes));char *text=NULL;ok(tididi_vtree_to_text(v,&text));
    TididiVtree *restored=NULL;TididiCircuit *loaded=NULL;ok(tididi_vtree_from_text(text,&restored));
    ok(tididi_from_bytes(restored,tididi_bytes_data(bytes),tididi_bytes_len(bytes),&loaded));CHECK(count(loaded)==6);
    TididiCircuit *bad=NULL;uint8_t corrupt[]={0,1,2};failure(tididi_from_bytes(restored,corrupt,3,&bad),TIDIDI_ERROR_CODE_INVALID_ARGUMENT);
    TididiNodeSizes *sizes=NULL;ok(tididi_node_sizes(f,&sizes));size_t pairs=0,nodes=0,sum=0;ok(tididi_size(f,&nodes,&pairs));
    for(size_t i=0;i<tididi_node_sizes_len(sizes);++i)sum+=tididi_node_sizes_data(sizes)[i].pairs;
    CHECK(sum==pairs);
    TididiCounter *counter=NULL;ok(tididi_counter(f,&counter));ok(tididi_circuit_free(f));tididi_vtree_free(v);
    uint64_t n=0;ok(tididi_counter_model_count(counter,&n,NULL));CHECK(n==6);
    int64_t observation[]={-1};ok(tididi_counter_observe(counter,observation,1));ok(tididi_counter_model_count(counter,&n,NULL));CHECK(n==2);
    int64_t invalid[]={1,4};failure(tididi_counter_observe(counter,invalid,2),TIDIDI_ERROR_CODE_INVALID_ARGUMENT);
    ok(tididi_counter_model_count(counter,&n,NULL));CHECK(n==2);
    ok(tididi_counter_clear(counter,1));ok(tididi_counter_model_count(counter,&n,NULL));CHECK(n==6);
    TididiCircuit *recovered=NULL;ok(tididi_counter_finish(counter,&recovered));CHECK(count(recovered)==6);
    failure(tididi_counter_model_count(counter,&n,NULL),TIDIDI_ERROR_CODE_INVALID_ARGUMENT);
    ok(tididi_counter_free(counter));ok(tididi_circuit_free(recovered));ok(tididi_circuit_free(loaded));tididi_vtree_free(restored);
    tididi_bytes_free(bytes);tididi_string_free(text);tididi_node_sizes_free(sizes);
}
typedef struct { TididiCircuit *circuit; TididiCircuit *spare; size_t calls; } CallbackState;
static double zero(void *state) { (void)state;return 0; }
static double leaf(void *data,uint32_t var,int8_t sign) {
    CallbackState *state=(CallbackState*)data;(void)var;state->calls++;
    TididiCircuit *out=NULL;
    failure(tididi_negate(state->circuit,&out,NULL),TIDIDI_ERROR_CODE_BORROW_CONFLICT);
    failure(tididi_circuit_free(state->circuit),TIDIDI_ERROR_CODE_BORROW_CONFLICT);
    TididiCircuit *inputs[] = {state->spare, state->circuit};
    failure(tididi_or_many(inputs, 2, &out, NULL), TIDIDI_ERROR_CODE_BORROW_CONFLICT);
    CHECK(out == NULL && !consumed(state->spare));
    return sign<0?2:1;
}
static double add(void *data,double a,double b){(void)data;return a+b;}
static double mul(void *data,double a,double b){(void)data;return a*b;}
static void evaluation(void) {
    TididiVtree *v=NULL;ok(tididi_vtree_balanced(3,&v));TididiCircuit *f=NULL,*rain=lit(v,1),*no=NULL;
    int64_t values[]={1,2};ok(tididi_clause(v,values,2,&f,NULL));ok(tididi_zero(v,&no));
    TididiWeight weights[]={{1,"4/5","1/5"},{2,"9/10","1/10"},{3,"3/5","2/5"}};
    char *mass=NULL,*ratio=NULL;ok(tididi_weighted_count(f,weights,3,&mass,NULL));CHECK(strcmp(mass,"7/25")==0);
    ok(tididi_weighted_ratio(rain,f,weights,3,&ratio,NULL));CHECK(strcmp(ratio,"5/7")==0);
    char *bad=NULL;failure(tididi_weighted_ratio(f,no,weights,3,&bad,NULL),TIDIDI_ERROR_CODE_INVALID_ARGUMENT);
    failure(tididi_weighted_count(f,weights,2,&bad,NULL),TIDIDI_ERROR_CODE_INVALID_ARGUMENT);
    CallbackState state={f,rain,0};TididiAlgebra algebra={&state,zero,leaf,add,mul};double result=0;
    ok(tididi_evaluate_f64(f,&algebra,&result,NULL));CHECK(result==6 && state.calls>0 && count(f)==6 && count(rain)==4);
    tididi_string_free(mass);tididi_string_free(ratio);ok(tididi_circuit_free(f));ok(tididi_circuit_free(rain));ok(tididi_circuit_free(no));tididi_vtree_free(v);
}

static void conditioning(void) {
    TididiVtree *vtree = NULL;
    ok(tididi_vtree_balanced(3, &vtree));
    TididiCircuit *a = lit(vtree, 1), *b = lit(vtree, 1), *repeated = NULL, *conflict = NULL;
    const int64_t repeat[] = {1,1}, opposite[] = {1,-1};
    ok(tididi_condition(a, repeat, 2, &repeated, NULL));
    ok(tididi_condition(b, opposite, 2, &conflict, NULL));
    CHECK(count(repeated) == 8 && count(conflict) == 0);
    CHECK(consumed(a) && consumed(b));
    ok(tididi_circuit_free(a)); ok(tididi_circuit_free(b));
    ok(tididi_circuit_free(repeated)); ok(tididi_circuit_free(conflict));
    tididi_vtree_free(vtree);
}

static void weighted_evaluator(void) {
    TididiVtree *v = NULL;
    ok(tididi_vtree_balanced(2, &v));
    TididiCircuit *f = NULL;
    const int64_t literals[] = {1, 2};
    ok(tididi_clause(v, literals, 2, &f, NULL));
    TididiWeight weights[] = {{1, "4/5", "1/5"}, {2, "9/10", "1/10"}};
    TididiEvaluator *e = NULL;
    failure(tididi_evaluator(f, weights, 1, &e), TIDIDI_ERROR_CODE_INVALID_ARGUMENT);
    CHECK(count(f) == 3 && e == NULL);
    ok(tididi_evaluator(f, weights, 2, &e));
    CHECK(consumed(f));
    char *value = NULL;
    ok(tididi_evaluator_value(e, &value, NULL));
    CHECK(strcmp(value, "7/25") == 0); tididi_string_free(value); value = NULL;
    const int64_t evidence[] = {-1};
    ok(tididi_evaluator_observe(e, evidence, 1));
    const int64_t invalid[] = {1, 3};
    failure(tididi_evaluator_observe(e, invalid, 2), TIDIDI_ERROR_CODE_INVALID_ARGUMENT);
    ok(tididi_evaluator_value(e, &value, NULL));
    CHECK(strcmp(value, "2/25") == 0); tididi_string_free(value); value = NULL;
    TididiWeight unit[] = {{1, "1", "1"}, {2, "1", "1"}};
    ok(tididi_evaluator_set_weights(e, unit, 2));
    ok(tididi_evaluator_value(e, &value, NULL));
    CHECK(strcmp(value, "1") == 0); tididi_string_free(value); value = NULL;
    ok(tididi_evaluator_clear(e, 1));
    ok(tididi_evaluator_clear_all(e));
    ok(tididi_evaluator_value(e, &value, NULL));
    CHECK(strcmp(value, "3") == 0); tididi_string_free(value); value = NULL;
    TididiCircuit *restored = NULL;
    ok(tididi_evaluator_finish(e, &restored));
    CHECK(count(restored) == 3);
    failure(tididi_evaluator_value(e, &value, NULL), TIDIDI_ERROR_CODE_INVALID_ARGUMENT);
    ok(tididi_evaluator_free(e)); ok(tididi_evaluator_free(NULL));
    ok(tididi_circuit_free(restored)); ok(tididi_circuit_free(f)); tididi_vtree_free(v);
}

int main(void) {
    weighted_evaluator();conditioning();ownership();multi_operand_preflight();queries();big_counts_and_rows();counters_and_persistence();evaluation();
    puts("C ownership, queries, exact arithmetic, callbacks and persistence passed.");return 0;
}
