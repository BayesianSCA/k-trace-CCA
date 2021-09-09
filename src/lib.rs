#![allow(unused)]

use pyo3::prelude::*;

mod butterfly_node;
mod constants;
mod graph;
mod kyber_msg;
mod reduce;

use graph::BPGraphError;
use graph::KyberRefINTTGraph;

#[pymodule]
fn ntt_bp(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<KyberRefINTTGraph>()?;
    m.add("BPGraphError", py.get_type::<BPGraphError>())?;

    Ok(())
}

/*
#[allow(dead_code)]
#[cfg(test)]
pub mod tests {
    use test_helpers::*;
    use crate::*;
    use crate::{BPGraph, Msg, Probability, VariableNode};
    use belief_propagation::macros::*;
    use std::time::SystemTime;

    #[test]
    fn test_ntt_butterfly() -> Result<(), &'static str> {
        let q = 41 as u16;
        let p = 1.0 / (4.0 * ((q - 1) as Probability));
        let puni = 1.0 / (q as Probability);
        let omega = 2;
        let mut uniform = NTTMsg::new_with_size(q as usize - 1);
        let mut measure0 = NTTMsg::new_with_size(q as usize - 1);
        let mut measure1 = NTTMsg::new_with_size(q as usize - 1);
        println!("Creating priors");
        for i in 0..q {
            measure0.insert(i, p);
            measure1.insert(i, p);
            uniform.insert(i, puni);
        }
        measure0.insert(0, 0.75);
        measure1.insert(0, 0.75);
        println!("Creating graph");
        //
        let mut g = BPGraph::<u16, NTTMsg>::new();
        //Variable nodes
        let mut vnodein0 = VariableNode::new();
        let mut vnodein1 = VariableNode::new();
        let mut vnodeout = VariableNode::new();
        measure0.normalize()?;
        measure1.normalize()?;
        uniform.normalize()?;
        vnodein0.set_prior(&measure0)?;
        vnodein1.set_prior(&measure1)?;
        vnodeout.set_prior(&uniform)?;

        let v0 = g.add_node("in0".to_string(), Box::new(vnodein0));
        let v1 = g.add_node("in1".to_string(), Box::new(vnodein1));
        let v2 = g.add_node("in2".to_string(), Box::new(vnodeout.clone()));
        let v3 = g.add_node("in3".to_string(), Box::new(vnodeout));

        //Factor nodes
        let b4 = g.add_node(
            "butterfly".to_string(),
            Box::new(NTTButterflyNode::new(omega, q)),
        );
        assert!(!g.is_valid());

        //Add edges
        g.add_edge(v0, b4)?;
        g.add_edge(v1, b4)?;
        g.add_edge(v2, b4)?;
        assert!(!g.is_valid());
        g.add_edge(v3, b4)?;

        g.initialize()?;
        g.propagate(20)?;
        //
        assert!(g.is_valid());

        let mut res = g.get_result(v3)?.unwrap();
        let mut sum = 0.0;
        for p in &res {
            sum += p.1;
        }

        let expected_0 = (3 as Probability / 4.0).powi(2) + 1.0 / ((16 * (q - 1)) as Probability);
        let expected_else = (1.0 - expected_0) / ((q - 1) as Probability);
        for (val, p) in res.iter_mut() {
            *p /= sum;
            if *val != 0 {
                assert!(f64::abs(*p - expected_else) < 0.0000001);
            } else {
                assert!(f64::abs(*p - expected_0) < 0.0000001);
            }
        }

        Ok(())
    }

    #[test]
    fn test_ntt_graph() -> Result<(), &'static str> {
        let iterations = 30;
        println!("Building graph...");
        let q = 3329;
        let mut g = ntt_graph(1, 1, q, vec![7])?;
        assert!(g.is_valid());
        println!(
            "Graph has {} nodes ({} factor nodes and {} variables nodes)",
            g.nodes_count(),
            g.factor_nodes_count(),
            g.variable_nodes_count()
        );
        println!("Propagating {} iteration with 1 threads..", iterations);
        let now = SystemTime::now();
        g.propagate_threaded(iterations, 2)?;
        match now.elapsed() {
            Ok(elapsed) => {
                let s = elapsed.as_secs();
                println!(
                    "Propagating took {} seconds (= {:.2} minutes)",
                    s,
                    s as f32 / 60.0
                );
            }
            Err(e) => {
                println!("Could not measure time: {:?}", e);
            }
        }
        let mut res = g.get_result(g.len() - 2)?.unwrap();
        let mut sum = 0.0;
        for (_, p) in &res {
            sum += p;
        }

        let expected_0 = (3 as Probability / 4.0).powi(2) + 1.0 / ((16 * (q - 1)) as Probability);
        let expected_else = (1.0 - expected_0) / ((q - 1) as Probability);

        for (val, p) in res.iter_mut() {
            *p /= sum;
            if *val != 0 {
                assert!(f64::abs(*p - expected_else) < 0.0000001);
            } else {
                assert!(f64::abs(*p - expected_0) < 0.0000001);
            }
        }

        Ok(())
    }

    #[test]
    fn test_kyber_ntt_graph() -> Result<(), &'static str> {
        let iterations = 30;
        println!("Building graph...");
        let mut g = ntt_graph_kyber(1, 1, vec![7])?;
        assert!(g.is_valid());
        println!(
            "Graph has {} nodes ({} factor nodes and {} variables nodes)",
            g.nodes_count(),
            g.factor_nodes_count(),
            g.variable_nodes_count()
        );
        println!("Propagating {} iteration with 1 thread..", iterations);
        let now = SystemTime::now();
        g.propagate(iterations)?;
        match now.elapsed() {
            Ok(elapsed) => {
                let s = elapsed.as_secs();
                println!(
                    "Propagating took {} seconds (= {:.2} minutes)",
                    s,
                    s as f32 / 60.0
                );
            }
            Err(e) => {
                println!("Could not measure time: {:?}", e);
            }
        }

        let mut res = g.get_result(g.len() - 2)?.unwrap();
        let mut sum = 0.0;
        let mut vmax = 0;
        for p in &res {
            sum += p.1;
            vmax = std::cmp::min(*p.0, vmax);
        }
        for (_, p) in res.iter_mut() {
            *p /= sum;
        }

        let q = 3329;
        let expected_0 = (3 as Probability / 4.0).powi(2) + 1.0 / ((16 * (q - 1)) as Probability);
        let expected_else = (1.0 - expected_0) / ((q - 1) as Probability);

        assert!(f64::abs(res[&0] - expected_0) < 0.001);

        Ok(())
    }

    #[test]
    fn test_kyber_intt_graph() -> Result<(), &'static str> {
        let iterations = 2;
        let q = 3329;
        println!("Building graph...");
        let mut g = intt_graph_kyber(1, 1, vec![7])?;
        assert!(g.is_valid());
        println!(
            "Graph has {} nodes ({} factor nodes and {} variables nodes)",
            g.nodes_count(),
            g.factor_nodes_count(),
            g.variable_nodes_count()
        );
        println!("Propagating {} iteration with 1 thread..", iterations);
        let now = SystemTime::now();
        g.propagate(iterations)?;
        match now.elapsed() {
            Ok(elapsed) => {
                let s = elapsed.as_secs();
                println!(
                    "Propagating took {} seconds (= {:.2} minutes)",
                    s,
                    s as f32 / 60.0
                );
            }
            Err(e) => {
                println!("Could not measure time: {:?}", e);
            }
        }

        let mut res = g.get_result(g.len() - 2)?.unwrap();
        let mut sum = 0.0;
        for p in &res {
            sum += p.1;
        }

        let expected_0 = (3 as Probability / 4.0).powi(2) + 1.0 / ((16 * (q - 1)) as Probability);
        let expected_else = (1.0 - expected_0) / ((q - 1) as Probability);

        for (val, p) in res.iter_mut() {
            *p /= sum;
            if *val != 0 {
                assert!(f64::abs(*p - expected_else) < 0.001);
            } else {
                println!("{}, {}", *p, expected_0);
                assert!(f64::abs(*p - expected_0) < 0.001);
            }
        }

        Ok(())
    }
}
*/
