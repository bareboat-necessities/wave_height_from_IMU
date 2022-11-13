mod plot;

#[macro_use]
extern crate rulinalg;
extern crate linearkalman;

use rulinalg::vector::Vector;
use linearkalman::{KalmanFilter, KalmanState, update_step, predict_step};

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
    let pos_integral_trans_variance: f64 = 100.0;
    let pos_integral_variance: f64 = 100.0;

    let b = Vector::new(vec![((1.0 / 6.0) * dt.powf(3.0)), (0.5 * dt.powf( 2.0)), dt]);

    let x0 = vector![0.0, 0.0, 0.0];
    let p0 = matrix![pos_integral_variance, 0.0, 0.0;
                                                  0.0, 1.0, 0.0;
                                                  0.0, 0.0, 1.0];
    let kf = KalmanFilter {
        // Process noise covariance
        q: matrix![pos_integral_trans_variance, 0.0, 0.0;
                                           0.0, 0.2, 0.0;
                                           0.0, 0.0, 0.1],
        // Measurement noise matrix
        r: matrix![pos_integral_variance],
        // Observation matrix
        h: matrix![1.0, 0.0, 0.0],
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
    for _t in 0..n_timesteps {
        data.push(Vector::new(vec![0.0]));
    }

    let acc_mean = -0.08; // TODO: calculate it

    // With no integral drift correction
    let mut predicted: Vec<KalmanState> = Vec::with_capacity(n_timesteps+1);
    let mut filtered: Vec<KalmanState> = Vec::with_capacity(n_timesteps);

    predicted.push(KalmanState { x: (kf.x0).clone(), p: (kf.p0).clone() });

    for k in 0..n_timesteps {
        filtered.push(update_step(&kf, &predicted[k], &data[k]));
        filtered[k].x = &filtered[k].x + &b * (acc[k] - acc_mean);
        predicted.push(predict_step(&kf, &filtered[k]));
    }

    let mut result_pos: Vec<f64> = Vec::new();
    //let mut result_vel: Vec<f64> = Vec::new();
    for t in 0..n_timesteps {
        result_pos.push(filtered[t].x[1]);
        //result_vel.push(filtered[t].x[2]);
    }

    let env = Env::new();
    let plot = Plot::new(&env);

    plot.plot(&time, &result_pos);
    plot.plot(&time, &pos);
    plot.grid(true);
    plot.xlabel("Time");
    plot.ylabel("Pos");
    plot.title("Est Pos vs Ref Pos");
    plot.set_ylim(-2.0, 2.0);
    plot.draw();
    plot.show();
}

