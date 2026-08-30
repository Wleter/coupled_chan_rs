use cc_problems::diatom_problems::hamiltonian_diatom_in_b_field;
use coupled_chan_cli::{
    ProgramExecutor,
    input::diatom_problems,
};

fn main() {
    ProgramExecutor::default()
        .set_hamiltonian_builder(hamiltonian_diatom_in_b_field)
        .set_calculation_specs(diatom_problems())
        .build();
}
