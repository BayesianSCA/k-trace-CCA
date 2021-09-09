use crate::constants::KYBER_Q;
use crate::kyber_msg::KyberMsg;
use crate::reduce::*;
use belief_propagation::{BPError, BPResult, Msg, NodeFunction, NodeIndex, Probability};

pub struct ButterflyNode {
    nodes: Vec<NodeIndex>,
    zeta: KyberINTTVal,
}

pub type KyberINTTVal = i16;

fn match_position(
    nodes: &Vec<NodeIndex>,
    inbox: &Vec<(NodeIndex, KyberMsg)>,
) -> BPResult<[usize; 4]> {
    if (nodes.len() != 4 || inbox.len() != 4) {
        return Err(BPError::new(
            "butterfly_node::match_position".to_owned(),
            "Needs 4 connections/messages.".to_owned(),
        ));
    }

    let mut t = [0; 4];
    for (i, (ind, _)) in inbox.iter().enumerate() {
        //match can't evaluate arrays as pattern (makes sense?)
        let index = if *ind == nodes[0] {
            0
        } else if *ind == nodes[1] {
            1
        } else if *ind == nodes[2] {
            2
        } else if *ind == nodes[3] {
            3
        } else {
            return Err(BPError::new(
                "butterfly_node::match_position".to_owned(),
                "Unexpected index".to_owned(),
            ));
        };
        t[index] = i;
    }
    Ok(t)
}

impl ButterflyNode {
    pub fn new(zeta: i16) -> Self {
        Self {
            nodes: Vec::new(),
            zeta: zeta,
        }
    }

    fn f_normal(&self, inbox: Vec<(NodeIndex, KyberMsg)>) -> BPResult<Vec<(NodeIndex, KyberMsg)>> {
        //Ins
        let ins = match_position(&self.nodes, &inbox)?;
        let in0 = &inbox[ins[0]].1;
        let in1 = &inbox[ins[1]].1;
        let mut in2 = inbox[ins[2]].1.clone();
        let mut in3 = inbox[ins[3]].1.clone();
        //Outs

        let in0neg = -in0.highest_negative();
        let in0pos = in0.highest_positive();
        let in1neg = -in1.highest_negative();
        let in1pos = in1.highest_positive();
        //msg0.set_size(in0pos as usize, -in0neg as usize);
        //msg1.set_size(in1pos as usize, -in1neg as usize);
        let mut msg0 = KyberMsg::new_with_size(in0pos as usize, -in0neg as usize);
        let mut msg1 = KyberMsg::new_with_size(in1pos as usize, -in1neg as usize);
        let mut msg2 = KyberMsg::new_with_size(
            in2.highest_positive() as usize,
            in2.highest_negative() as usize,
        );
        let mut msg3 = KyberMsg::new_with_size(
            in3.highest_positive() as usize,
            in3.highest_negative() as usize,
        );

        for b in in1neg..=in1pos {
            for a in in0neg..=in0pos {
                let ap = barrett_reduce(a + b);
                let bp = fqmul(self.zeta, b - a); //montg((a-b)*self.zeta as i32)

                let partial_pab = in0[a] * in1[b];
                let partial_papbp = in2[ap] * in3[bp];

                msg0[a] += partial_papbp * in1[b]; // = in1[b]*in2[ap]*in3[bp]
                msg1[b] += partial_papbp * in0[a];
                msg2[ap] += partial_pab * in3[bp];
                msg3[bp] += partial_pab * in2[ap];
            }
        }

        Ok(vec![
            (self.nodes[0], msg0),
            (self.nodes[1], msg1),
            (self.nodes[2], msg2),
            (self.nodes[3], msg3),
        ])
    }
}

impl NodeFunction<KyberINTTVal, KyberMsg> for ButterflyNode {
    fn node_function(
        &mut self,
        inbox: Vec<(NodeIndex, KyberMsg)>,
    ) -> BPResult<Vec<(NodeIndex, KyberMsg)>> {
        Ok(self.f_normal(inbox)?)
    }
    fn is_factor(&self) -> bool {
        true
    }
    fn get_prior(&self) -> Option<KyberMsg> {
        None
    }
    fn number_inputs(&self) -> Option<usize> {
        Some(4)
    }
    fn initialize(&mut self, connections: Vec<NodeIndex>) -> BPResult<()> {
        if connections.len() != 4 {
            Err(BPError::new("ButterflyNode::initialize".to_owned(), "Wrong number of connections. ButterflyNode needs to be initialized with 4 connections".to_owned()))
        } else {
            self.nodes = connections;
            Ok(())
        }
    }
    fn is_ready(&self, recv_msg: &Vec<(NodeIndex, KyberMsg)>, step: usize) -> BPResult<bool> {
        Ok(recv_msg.len() == self.nodes.len())
    }
    fn reset(&mut self) -> BPResult<()> {
        Ok(())
    }
}
