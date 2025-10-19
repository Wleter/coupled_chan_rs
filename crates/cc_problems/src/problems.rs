use indicatif::ProgressStyle;

pub fn default_progress() -> ProgressStyle {
    ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] ({eta})")
        .unwrap()
        .progress_chars("#>-")
}
