mod libs;

use libs::{Env, Plot};

fn read_data(path: &str) -> (Vec<f32>, Vec<f32>) {
    let mut x = Vec::new();
    let mut y = Vec::new();

    let rdr = csv::Reader::from_path(path);
    for result in rdr.expect("Missing data file?").records().into_iter()  {
        let record = result.expect("Missing record?");
        x.push(record.get(0).expect("Missing field 0").trim().parse::<f32>().unwrap());
        y.push(record.get(2).expect("Missing field 2").trim().parse::<f32>().unwrap());
    }
    
    (x, y)
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
