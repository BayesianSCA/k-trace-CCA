use crate::butterfly_node::ButterflyNode;
use crate::kyber_msg::KyberMsg;
use belief_propagation::variable_node::{InputNeed, VariableNode};
use belief_propagation::{hashmap_to_distribution, BPError, BPGraph, BPResult, Msg, Probability};
use crossbeam;
use pyo3::create_exception;
use pyo3::prelude::*;
use pyo3::{exceptions, PyResult};
use std::collections::HashMap;

create_exception!(pybpgraph, BPGraphError, pyo3::exceptions::PyException);

#[derive(Debug)]
pub struct KyberRefINTTGraphError {
    desc: String,
}

impl KyberRefINTTGraphError {
    pub fn new(desc: String) -> Self {
        KyberRefINTTGraphError { desc: desc }
    }
    pub fn from_bp(err: BPError) -> Self {
        KyberRefINTTGraphError {
            desc: err.to_string(),
        }
    }
}

impl std::fmt::Display for KyberRefINTTGraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "An error occured in belief_propagation: {}", self.desc)
    }
}

impl std::error::Error for KyberRefINTTGraphError {}

impl std::convert::From<BPError> for KyberRefINTTGraphError {
    fn from(err: BPError) -> KyberRefINTTGraphError {
        KyberRefINTTGraphError::from_bp(err)
    }
}

impl std::convert::From<KyberRefINTTGraphError> for PyErr {
    fn from(err: KyberRefINTTGraphError) -> PyErr {
        BPGraphError::new_err(err.to_string())
    }
}

#[pyclass]
pub struct KyberRefINTTGraph {
    g: BPGraph<i16, KyberMsg>,
}

#[pymethods]
impl KyberRefINTTGraph {
    #[new]
    fn new() -> Self {
        Self { g: BPGraph::new() }
    }
    fn set_check_validity(&mut self, value: bool) {
        self.g.set_check_validity(value);
    }

    fn add_butterfly_node(&mut self, name: String, omega: i16) -> usize {
        let n = ButterflyNode::new(omega);
        let idx = self.g.add_node(name, Box::new(n));
        idx
    }

    fn add_threaded_var_node(
        &mut self,
        name: String,
        prior: Option<HashMap<i16, f64>>,
        val_min: i16,
        val_max: i16,
        self_start: bool,
    ) -> PyResult<usize> {
        let mut n = VariableNode::new();

        // https://doc.rust-lang.org/rust-by-example/flow_control/if_let.html
        if let Some(prior) = prior {
            let mut prior_msg = KyberMsg::new_with_size(val_max as usize, (-val_min) as usize);
            for (v, p) in prior {
                prior_msg.insert(v, p);
            }
            prior_msg
                .normalize()
                .map_err(|e| KyberRefINTTGraphError::from_bp(e))?;
            n.set_prior(&prior_msg)
                .map_err(|e| KyberRefINTTGraphError::from_bp(e))?;
        }
        if self_start {
            n.set_input_need(InputNeed::AlwaysExceptFirst);
        } else {
            n.set_input_need(InputNeed::Never);
        }
        let idx = self.g.add_node(name, Box::new(n));
        Ok(idx)
    }

    fn ini(&mut self) -> PyResult<()> {
        self.g
            .initialize()
            .map_err(|e| KyberRefINTTGraphError::from_bp(e))?;
        Ok(())
    }
    fn propagate(&mut self, steps: usize, threads: u32) -> PyResult<()> {
        if threads <= 0 {
            Err(KyberRefINTTGraphError::new(
                "Cannot work with less than 1 thread".to_owned(),
            ))?;
        } else if threads == 1 {
            self.g
                .propagate(steps)
                .map_err(|e| KyberRefINTTGraphError::from_bp(e))?;
        } else {
            self.g
                .propagate_threaded(steps, threads)
                .map_err(|e| KyberRefINTTGraphError::from_bp(e))?;
        }
        Ok(())
    }
    fn get_results(
        &self,
        nodes: Vec<usize>,
        thread_count: usize,
    ) -> PyResult<HashMap<usize, Option<(HashMap<i16, Probability>, f64)>>> {
        let res = fetch_results_parallel(&self.g, nodes, thread_count)
            .map_err(|e| KyberRefINTTGraphError::from_bp(e))?;
        Ok(res)
    }
    fn get_result(&self, node: usize) -> PyResult<Option<HashMap<i16, Probability>>> {
        let res = self
            .g
            .get_result(node)
            .map_err(|e| KyberRefINTTGraphError::from_bp(e))?;
        Ok(res)
    }
    fn add_edge(&mut self, from: usize, to: usize) -> PyResult<()> {
        self.g
            .add_edge(from, to)
            .map_err(|e| KyberRefINTTGraphError::from_bp(e))?;
        Ok(())
    }
}

fn fetch_results_parallel(
    g: &BPGraph<i16, KyberMsg>,
    nodes: Vec<usize>,
    thread_count: usize,
) -> BPResult<HashMap<usize, Option<(HashMap<i16, Probability>, f64)>>> {
    crossbeam::scope(
        |scope| -> BPResult<HashMap<usize, Option<(HashMap<i16, Probability>, f64)>>> {
            let nodes_per_thread = nodes.len() / thread_count;
            let mut results = HashMap::new();
            let mut handles = Vec::new();
            for nodes_list in nodes.chunks(nodes_per_thread) {
                handles.push(scope.spawn(
                    move |_| -> Vec<(usize, BPResult<Option<(HashMap<i16, Probability>, f64)>>)> {
                        let mut tr_results = Vec::new();
                        for node in nodes_list {
                            let mut res = g.get_result(*node).map(|res| match res {
                                Some(mut r) => {
                                    hashmap_to_distribution(&mut r);
                                    let ent = calc_entropy(&r);
                                    if ent.is_nan() {
                                        println!("{:?}", r);
                                        panic!("");
                                    }
                                    Some((r, ent))
                                }
                                None => None,
                            });
                            tr_results.push((*node, res));
                        }
                        tr_results
                    },
                ));
            }
            for h in handles {
                let tr = h.join().expect("Joining threads failed in get_results.");
                for (node, res) in tr.into_iter() {
                    results.insert(node, res?);
                }
            }
            Ok(results)
        },
    )
    .expect("Scoped threading failed.")
}

fn calc_entropy(probs: &HashMap<i16, Probability>) -> f64 {
    -probs
        .iter()
        .map(|(_, p)| {
            let r = p * p.log2();
            if r.is_nan() {
                0.0
            } else {
                r
            }
        })
        .sum::<f64>()
}
