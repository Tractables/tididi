#include "tididi.h"
#include <cstdlib>
#include <memory>
#include <iostream>

static void check(TididiError *error) {
    if (error) { std::cerr << tididi_error_message(error) << '\n'; tididi_error_free(error); std::abort(); }
}
struct CircuitDeleter { void operator()(TididiCircuit *f) const { check(tididi_circuit_free(f)); } };
int main() {
    TididiVtree *raw_vtree=nullptr;check(tididi_vtree_balanced(3,&raw_vtree));
    std::unique_ptr<TididiVtree,decltype(&tididi_vtree_free)> vtree(raw_vtree,tididi_vtree_free);
    TididiCircuit *raw=nullptr;check(tididi_literal(vtree.get(),1,&raw,nullptr));
    std::unique_ptr<TididiCircuit,CircuitDeleter> circuit(raw);
    uint64_t models=0;check(tididi_model_count(circuit.get(),&models,nullptr));
    if(models!=4) return 1;
    std::cout << "C++ header and RAII integration passed.\n";
}
