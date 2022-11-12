mod plot;

#[macro_use]
extern crate rulinalg;
extern crate linearkalman;

use rulinalg::vector::Vector;
use linearkalman::{KalmanFilter};

use csv::StringRecord;
use plot::{Env, Plot};

fn read_data(path: &str) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut x = Vec::new();
    let mut y1 = Vec::new();
    let mut y2 = Vec::new();
    let mut y3 = Vec::new();
    let rdr = csv::Reader::from_path(path);
    for result in rdr.expect("Missing data file?").records().into_iter()  {
        let record = result.expect("Missing record?");
        x.push(get_f64(&record, 0));
        y1.push(get_f64(&record, 1));
        y2.push(get_f64(&record, 2));
        y3.push(get_f64(&record, 3));
    }
    (x, y1, y2, y3)
}

fn get_f64(record: &StringRecord, index: usize) -> f64 {
    record.get(index).expect(&*format!("Missing field {index}")).trim().parse::<f64>().unwrap()
}

fn main() {
    let (time, acc, pos, _ /*vel*/) = read_data("../test-data.txt");

    let dt = time[1] - time[0];
    let acc_x_variance = 0.0007;

    let x0 = vector![0.0, 0.0, acc[0]];
    let p0 = matrix![0.0, 0.0, 0.0;
                                0.0, 0.0, 0.0;
                                0.0, 0.0, acc_x_variance];
    let kf = KalmanFilter {
        // Process noise covariance
        q: matrix![0.1, 0.0, 0.0;
                   0.0, 0.2, 0.0;
                   0.0, 0.0, 0.0001],
        // Measurement noise matrix
        r: matrix![acc_x_variance],
        // Observation matrix
        h: matrix![0.0, 0.0, 1.0],
        // State transition matrix
        f: matrix![1.0,  dt,  dt.powf(2.0)/2.0;
                   0.0, 1.0,                dt;
                   0.0, 0.0,               1.0],
        // Initial guess for state mean at time 0
        x0: x0,
        // Initial guess for state covariance at time 0
        p0: p0,
    };

    let n_timesteps = time.len();

    let mut data: Vec<Vector<f64>> = Vec::new();
    for t in 0..n_timesteps {
        data.push(Vector::new(vec![acc[t]]));
    }

    // With no integral drift correction
    let (filtered, _ /*predicted*/) = kf.filter(&data);

    let mut result_pos: Vec<f64> = Vec::new();
    //let mut result_vel: Vec<f64> = Vec::new();
    for t in 0..n_timesteps {
        result_pos.push(filtered[t].x[0]);
        //result_vel.push(filtered[t].x[1]);
    }

    let env = Env::new();
    let plot = Plot::new(&env);

    plot.plot(&time, &result_pos);
    plot.plot(&time, &pos);
    plot.grid(true);
    plot.xlabel("Time");
    plot.ylabel("Pos");
    plot.title("Est Pos vs Ref Pos");
    plot.set_ylim(-20.0, 2.0);
    plot.draw();
    plot.show();
}

