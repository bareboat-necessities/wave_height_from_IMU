extern crate cpython;

use cpython::Python;
use cpython::{PyModule, PyTuple};

pub struct Env {
    gil: cpython::GILGuard,
}

impl Env {
    pub fn new() -> Env {
        Env { gil: Python::acquire_gil() }
    }
}

pub struct Plot<'p> {
    py: Python<'p>,
    plt: PyModule,
}

impl<'p> Plot<'p> {
    pub fn new(env: &Env) -> Plot {
        let py = env.gil.python();
        let plt = PyModule::import(py, "matplotlib.pyplot").unwrap();

        Plot { py: py, plt: plt }
    }

    pub fn show(&self) {
        let _ = self.plt.call(self.py, "show", PyTuple::empty(self.py), None).unwrap();
    }

    pub fn draw(&self) {
        let _ = self.plt.call(self.py, "draw", PyTuple::empty(self.py), None).unwrap();
    }

    pub fn set_ylim(&self, ymin: f64, ymax: f64) {
        let _ = self.plt.call(self.py, "ylim", (ymin, ymax), None).unwrap();
    }

    pub fn plot(&self, x: &[f64], y: &[f64]) {
        assert_eq!(x.len(), y.len());
        let _ = self.plt.call(self.py, "plot", (x, y), None).unwrap();
    }

    pub fn title(&self, title: &str) {
        let _ = self.plt.call(self.py, "title", (title,), None).unwrap();
    }

    pub fn xlabel(&self, label: &str) {
        let _ = self.plt.call(self.py, "xlabel", (label,), None).unwrap();
    }

    pub fn ylabel(&self, label: &str) {
        let _ = self.plt.call(self.py, "ylabel", (label,), None).unwrap();
    }

    pub fn grid(&self, grid: bool) {
        let _ = self.plt.call(self.py, "grid", (grid,), None).unwrap();
    }
}
