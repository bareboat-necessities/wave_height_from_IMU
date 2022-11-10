mod libs;

use libs::{Env, Plot};

fn main() {
    let mut x = Vec::new();
    let mut y = Vec::new();

    for i in 1..100 {
        x.push(i as f32);
        y.push(100.0 - i as f32);
    }

    let env = Env::new();
    let plot = Plot::new(&env);

    plot.plot(&x, &y);
    plot.grid(true);
    plot.xlabel("X");
    plot.ylabel("Y");
    plot.title("Simple Plot");

    plot.show();
}
