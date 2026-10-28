use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;

use anyhow::Result;
use cc_problems::problems::AvailableProblems;

#[derive(clap::Args, Debug)]
pub struct SchemeArgs {
    /// input file path
    #[arg(short, long)]
    problem_name: Box<str>,

    /// input file path
    #[arg(short, long)]
    save_path: PathBuf,
}

impl SchemeArgs {
    pub fn generate_scheme(&self, problems: AvailableProblems) -> Result<()> {
        let scheme = problems.scheme_for(&self.problem_name)?;

        let mut path = std::env::current_dir()?;
        path.push(&self.save_path);

        let file = File::create(&path)?;
        let writer = BufWriter::new(file);

        serde_json::to_writer_pretty(writer, &scheme)?;

        Ok(())
    }
}
