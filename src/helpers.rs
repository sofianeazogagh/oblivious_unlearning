pub use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::io::Write;

use crate::VERBOSE;

pub fn log(filepath: &str, message: &str) {
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(filepath)
        .unwrap();
    writeln!(file, "{}", message).unwrap();
}

// Macro helper pour créer une barre de progression conditionnelle
#[macro_export]
macro_rules! create_progress_bar {
    ($mp:expr, $len:expr, $style:expr, $msg:expr) => {
        if VERBOSE {
            let pb = $mp.add(ProgressBar::new($len));
            pb.set_style($style);
            pb.set_message($msg);
            Some(pb)
        } else {
            None
        }
    };
}

// Macro helper pour incrémenter la barre de progression
#[macro_export]
macro_rules! inc_progress {
    ($pb:expr) => {
        if let Some(pb) = $pb {
            pb.inc(1);
        }
    };
}

// Macro helper pour terminer la barre de progression
#[macro_export]
macro_rules! finish_progress {
    ($pb:expr) => {
        if let Some(pb) = $pb {
            pb.finish();
        }
    };
}

pub fn make_pb(mp: &MultiProgress, len: u64, msg: impl Into<String>) -> Option<ProgressBar> {
    create_progress_bar!(
        mp,
        len,
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] Tree {msg}: {bar:40.green} {pos:>7}/{len:7} {percent}%")
            .unwrap()
            .progress_chars("█░"),
        msg.into()
    )
}
