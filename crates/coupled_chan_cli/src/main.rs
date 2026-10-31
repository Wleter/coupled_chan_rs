use cc_problems::problems::available_problems;
use coupled_chan_cli::CoupledChanCli;

fn main() -> anyhow::Result<()> {
    CoupledChanCli::parse().execute(available_problems())
}
