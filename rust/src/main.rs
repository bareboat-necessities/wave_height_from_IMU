mod libs;

use csv::StringRecord;
use libs::{Env, Plot};

fn read_data(path: &str) -> (Vec<f32>, Vec<f32>) {
    let mut x = Vec::new();
    let mut y = Vec::new();

    let rdr = csv::Reader::from_path(path);
    for result in rdr.expect("Missing data file?").records().into_iter()  {
        let record = result.expect("Missing record?");
        x.push(get_as(&record, 0));
        y.push(get_as(&record, 2));
    }

    (x, y)
}

fn get_as(record: &StringRecord, index: usize) -> f32 {
    record.get(index).expect(&*format!("Missing field {index}")).trim().parse::<f32>().unwrap()
}

fn plot_data(x: &Vec<f32>, y: &Vec<f32>) {
    let env = Env::new();
    let plot = Plot::new(&env);

    plot.plot(&x, &y);
    plot.grid(true);
    plot.xlabel("X");
    plot.ylabel("Y");
    plot.title("Simple Plot");

    plot.show();
}

fn main() {
    let (x, y) = read_data("../trochoidal_wave.txt");
    plot_data(&x, &y);
}
