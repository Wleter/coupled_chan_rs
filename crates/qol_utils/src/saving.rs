use anyhow::Result;
use serde::Serialize;
use std::{
    fmt::LowerExp,
    fs::{File, create_dir_all},
    io::Write,
    path::Path,
};

pub fn save_data(filename: &str, header: &str, data: &[Vec<impl LowerExp>]) -> Result<()> {
    let n = data.first().unwrap().len();
    for values in data {
        assert!(values.len() == n, "Same length data allowed only")
    }

    let mut path = std::env::current_dir().unwrap();
    path.push("data");
    path.push(filename);
    path.set_extension("dat");
    let filepath = path.parent().unwrap();

    let mut buf = header.to_string();
    for i in 0..n {
        let line = data.iter().fold(String::new(), |s, val| s + &format!("\t{:e}", val[i]));

        buf.push_str(&format!("\n{}", line.trim()));
    }

    if !Path::new(filepath).exists() {
        create_dir_all(filepath)?;
        println!("created path {}", filepath.display());
    }

    let mut file = File::create(&path)?;
    file.write_all(buf.as_bytes())?;

    println!("saved data on {}", path.display());
    Ok(())
}

pub fn save_serialize(filename: &str, data: &impl Serialize) -> Result<()> {
    let mut path = std::env::current_dir().unwrap();
    path.push("data");
    path.push(filename);
    path.set_extension("json");
    let filepath = path.parent().unwrap();

    let buf = serde_json::to_string(data).unwrap();

    if !Path::new(filepath).exists() {
        create_dir_all(filepath)?;
        println!("created path {}", filepath.display());
    }

    let mut file = File::create(&path)?;
    file.write_all(buf.as_bytes())?;

    println!("saved data on {}", path.display());
    Ok(())
}

pub fn save_spectrum(
    filename: &str,
    header: &str,
    parameter: &[impl LowerExp],
    spectra: &[Vec<impl LowerExp>],
) -> Result<()> {
    assert_eq!(
        parameter.len(),
        spectra.len(),
        "parameters and energies have to have the same length"
    );

    let mut path = std::env::current_dir().unwrap();
    path.push("data");
    path.push(filename);
    path.set_extension("dat");
    let filepath = path.parent().unwrap();

    let mut buf = header.to_string();

    for (p, e) in parameter.iter().zip(spectra.iter()) {
        let line = e.iter().fold(format!("{:e}", p), |s, val| s + &format!("\t{:e}", val));

        buf.push_str(&format!("\n{line}"))
    }

    if !Path::new(filepath).exists() {
        create_dir_all(filepath)?;
        println!("created path {}", filepath.display());
    }

    let mut file = File::create(&path)?;
    file.write_all(buf.as_bytes())?;

    println!("saved data on {}", path.display());
    Ok(())
}
