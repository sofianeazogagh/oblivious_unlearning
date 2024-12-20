use std::io::Write;

pub fn log(filepath: &str, message: &str) {
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(filepath)
        .unwrap();
    writeln!(file, "{}", message).unwrap();
}
