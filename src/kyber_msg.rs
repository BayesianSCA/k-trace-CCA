use belief_propagation::{BPError, BPResult, Msg, Probability};

#[derive(Clone, Debug)]
pub struct KyberMsg {
    pub data: Vec<Probability>,
    //Index of the last positive element i.e. the number of positive elements minus 1
    pub last_positive_element: usize, //This is bad style but probably faster than using an option?
    pub has_size: bool,
}

pub struct KyberMsgIterator<'a> {
    msg: &'a KyberMsg,
    index: i16,
}

pub struct KyberMsgIntoIterator {
    msg: KyberMsg,
    index: i16,
}

impl IntoIterator for KyberMsg {
    type Item = (i16, Probability);
    type IntoIter = KyberMsgIntoIterator;

    fn into_iter(self) -> Self::IntoIter {
        let index = -(self.data.len() as i16 - (self.last_positive_element as i16)) + 1;
        Self::IntoIter {
            msg: self,
            index: index,
        }
    }
}

impl Iterator for KyberMsgIntoIterator {
    type Item = (i16, Probability);
    fn next(&mut self) -> Option<Self::Item> {
        if self.index < 0 {
            if self.index.abs() as usize + self.msg.last_positive_element < self.msg.data.len() {
                let res = Some((self.index, self.msg[self.index]));
                self.index += 1;
                res
            } else {
                None
            }
        } else {
            if (self.index as usize) <= self.msg.last_positive_element {
                let res = Some((self.index, self.msg[self.index]));
                self.index += 1;
                res
            } else {
                None
            }
        }
    }
}

impl<'a> IntoIterator for &'a KyberMsg {
    type Item = (i16, Probability);
    type IntoIter = KyberMsgIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        KyberMsgIterator {
            msg: self,
            index: -(self.data.len() as i16 - self.last_positive_element as i16) + 1,
        }
    }
}

impl<'a> Iterator for KyberMsgIterator<'a> {
    type Item = (i16, Probability);
    fn next(&mut self) -> Option<Self::Item> {
        println!("{}", self.index);
        if self.index < 0 {
            if self.index.abs() as usize + self.msg.last_positive_element < self.msg.data.len() {
                self.index += 1;
                Some((self.index, self.msg[self.index]))
            } else {
                None
            }
        } else {
            if (self.index as usize) <= self.msg.last_positive_element {
                self.index += 1;
                Some((self.index, self.msg[self.index]))
            } else {
                None
            }
        }
    }
}

impl std::ops::Index<i16> for KyberMsg {
    type Output = Probability;
    fn index(&self, val: i16) -> &Self::Output {
        let add = ((val >> 15) & 1) as usize * self.last_positive_element;
        &self.data[add + val.abs() as usize]
    }
}
impl std::ops::IndexMut<i16> for KyberMsg {
    fn index_mut(&mut self, val: i16) -> &mut Self::Output {
        let add = ((val >> 15) & 1) as usize * self.last_positive_element;
        &mut self.data[add + val.abs() as usize]
    }
}

impl KyberMsg {
    pub fn set_size(
        &mut self,
        mut highest_pos: usize,
        mut highest_neg: usize,
    ) -> Result<(), &'static str> {
        if self.has_size
            && self.highest_negative() as usize >= highest_pos
            && self.highest_negative() as usize >= highest_neg
        {
            return Err("Nothing to do");
        }
        if self.has_size {
            highest_pos = std::cmp::max(self.highest_positive() as usize, highest_pos);
            highest_neg = std::cmp::max(self.highest_negative() as usize, highest_neg);
        }
        self.data
            .resize(highest_pos + highest_neg + 1, 0 as Probability);
        self.last_positive_element = highest_pos;
        self.has_size = true;
        Ok(())
    }

    pub fn highest_positive(&self) -> i16 {
        self.last_positive_element as i16
    }

    pub fn highest_negative(&self) -> i16 {
        (self.data.len() as i16 - self.last_positive_element as i16 - 1) as i16
    }

    pub fn get_data_clone(&self) -> Vec<Probability> {
        self.data.clone()
    }

    pub fn new_with_size(highest_pos: usize, highest_neg: usize) -> Self {
        let mut data = Vec::new();
        data.resize(highest_pos + highest_neg + 1, 0 as Probability);
        KyberMsg {
            data: data,
            last_positive_element: highest_pos,
            has_size: true,
        }
    }
    pub fn size(&self) -> (usize, usize) {
        (
            self.highest_positive() as usize + 1,
            self.highest_negative() as usize,
        )
    }
    pub fn check_is_valid_nan(&self) -> bool {
        for p in &self.data {
            if p.is_nan() {
                return false;
            }
        }
        return true;
    }
    pub fn to_distribution(&mut self) {
        let sum: f64 = self.data.iter().sum();
        if sum <= 0.0 || sum.is_nan() {
            panic!("Could not find valid sum");
        }
        self.data.iter_mut().for_each(|p| *p /= sum);
    }
}

impl Msg<i16> for KyberMsg {
    fn new() -> Self {
        KyberMsg {
            data: Vec::new(),
            last_positive_element: 0,
            has_size: false,
        }
    }
    fn get(&self, value: i16) -> Option<Probability> {
        if (value >= 0 && value > self.highest_positive()) || value.abs() > self.highest_negative()
        {
            None
        } else {
            Some(self[value])
        }
    }
    fn get_mut(&mut self, value: i16) -> Option<&mut Probability> {
        if (value >= 0 && value > self.highest_positive()) || value.abs() > self.highest_negative()
        {
            None
        } else {
            Some(&mut self[value])
        }
    }
    fn insert(&mut self, value: i16, p: Probability) {
        let add = ((value >> 15) & 1) as usize * self.last_positive_element;
        let idx = add + value.abs() as usize;
        if idx >= self.data.len() || (value >= 0 && value > self.last_positive_element as i16) {
            if value >= 0 {
                if idx > self.last_positive_element + 1 {
                    println!(
                        "Index is {} and last_positive_element is {}",
                        idx, self.last_positive_element
                    );
                    panic!("Cannot enlarge a KyberMsg by more than one positive element while inserting (not implemented).");
                }
                self.data.insert(value as usize, p);
                self.last_positive_element += 1;
            } else {
                self.data.resize(idx + 1, 0 as Probability);
                self.data[idx] = p;
            }
        } else {
            self.data[idx] = p;
        }
    }
    fn normalize(&mut self) -> BPResult<()> {
        let max = {
            *self
                .data
                .iter()
                .max_by(|p0, p1| p0.partial_cmp(p1).unwrap_or(std::cmp::Ordering::Less))
                .unwrap_or(&f64::NAN)
        };
        if max == 0 as Probability || max.is_nan() {
            return Err(BPError::new(
                "KyberMsg::normalize".to_owned(),
                "Did not find a useful value to normalize by".to_owned(),
            )
            .attach_debug_object("max (maximal probability value in the message)", max));
        }
        self.data.iter_mut().for_each(|p| *p /= max);
        Ok(())
    }

    fn is_valid(&self) -> bool {
        self.data
            .iter()
            .all(|p| !p.is_nan() && *p >= 0.0 && *p <= 1.0)
    }
    fn mult_msg(&mut self, other: &Self) {
        self.data
            .iter_mut()
            .zip(other.data.iter())
            .for_each(|(selfp, otherp)| *selfp *= otherp);
    }
}
