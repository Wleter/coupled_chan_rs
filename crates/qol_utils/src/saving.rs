use anyhow::{Result, anyhow};
use serde::Serialize;
use std::{
    fmt::LowerExp,
    fs::{File, OpenOptions, create_dir_all},
    io::{BufWriter, Write},
    path::Path,
    sync::mpsc::{Sender, channel},
    thread::JoinHandle,
};

pub struct DataSaver<D: Send> {
    tx: Option<Sender<D>>,
    handle: Option<JoinHandle<Result<()>>>,
}

impl<D: Send + Sync + 'static> DataSaver<D> {
    pub fn new<F: Format<D> + 'static>(filename: &str, formatter: F, file_access: FileAccess) -> Result<Self> {
        let (tx, rx) = channel();

        let mut path = std::env::current_dir()?;
        path.push(filename);
        let filepath = path.parent().ok_or(anyhow!("Could not get parent path"))?;

        if !Path::new(&filepath).exists() {
            create_dir_all(filepath)?;
            println!("created path {}", filepath.display());
        }

        let file = match file_access {
            FileAccess::Append => OpenOptions::new().append(true).create(true).open(&path)?,
            FileAccess::Create => File::create(&path)?,
        };

        let filepath = path.to_str().unwrap().to_string();
        let handle = std::thread::spawn(move || {
            let mut writer = BufWriter::new(file);
            if let Some(header) = formatter.header() {
                writeln!(writer, "{header}")?;
            }

            for result in rx.iter() {
                let formatted = formatter.format_data(&result);
                writeln!(writer, "{formatted}")?;
            }

            println!("Successfully saved {filepath}");

            Ok(())
        });

        Ok(Self {
            tx: Some(tx),
            handle: Some(handle),
        })
    }

    pub fn send(&self, data: D) -> Result<()> {
        self.tx.as_ref().unwrap().send(data)?;

        Ok(())
    }
}

impl<D: Send> Drop for DataSaver<D> {
    fn drop(&mut self) {
        if let Some(tx) = self.tx.take() {
            drop(tx);
        };
        if let Some(handle) = self.handle.take() {
            match handle.join() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => eprintln!("Thread returned error: {e:?}"),
                Err(e) => eprintln!("Thread join panicked: {e:?}"),
            }
        }
    }
}

pub enum FileAccess {
    Append,
    Create,
}

pub trait Format<D: ?Sized>: Send + Sync {
    fn header(&self) -> Option<&str> {
        None
    }

    fn format_data(&self, data: &D) -> String;
}

pub struct JsonFormat;

impl<D: Serialize> Format<D> for JsonFormat {
    fn format_data(&self, data: &D) -> String {
        serde_json::to_string(data).unwrap()
    }
}

pub struct DatFormat {
    header: String,
}

impl DatFormat {
    pub fn new(header: &str) -> Self {
        Self {
            header: header.to_string(),
        }
    }
}

impl<D: LowerExp> Format<[D]> for DatFormat {
    fn header(&self) -> Option<&str> {
        Some(&self.header)
    }

    fn format_data(&self, data: &[D]) -> String {
        use std::fmt::Write;

        let mut out = String::new();
        for x in data {
            // write! avoids intermediate String creation from format!
            write!(out, "{x:e},").unwrap();
        }
        out
    }
}

impl<D: LowerExp> Format<(D, Vec<D>)> for DatFormat {
    fn header(&self) -> Option<&str> {
        Some(&self.header)
    }

    fn format_data(&self, data: &(D, Vec<D>)) -> String {
        use std::fmt::Write;

        let mut out = format!("{:e}", data.0);
        for x in &data.1 {
            // write! avoids intermediate String creation from format!
            write!(out, "{x:e},").unwrap();
        }
        out
    }
}
