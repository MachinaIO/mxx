//! The circuits of every scheme and the depth a scheme evaluates.

/// A circuit of two-input gates in topological order, the circuits of every scheme.
///
/// Wires `[0, input_count)` are the inputs. Gate `i` is wire `input_count + i` and applies the
/// scheme's gate to the two earlier wires it names. The output may be any wire.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Circuit {
    input_count: usize,
    gates: Vec<[usize; 2]>,
    output: usize,
}

impl Circuit {
    /// A circuit whose gates name only earlier wires and whose output exists.
    pub fn new(input_count: usize, gates: Vec<[usize; 2]>, output: usize) -> Option<Self> {
        let topological = gates
            .iter()
            .enumerate()
            .all(|(gate, operands)| operands.iter().all(|&wire| wire < input_count + gate));
        (topological && output < input_count + gates.len()).then_some(Self {
            input_count,
            gates,
            output,
        })
    }

    /// The number of input wires.
    pub fn input_count(&self) -> usize {
        self.input_count
    }

    /// The number of gates on the longest path to the output.
    pub fn depth(&self) -> usize {
        let mut depths = vec![0; self.input_count];
        for [x, y] in &self.gates {
            depths.push(depths[*x].max(depths[*y]) + 1);
        }
        depths[self.output]
    }

    /// The output wire's value on `inputs`, with `gate` for each gate.
    pub fn run<T: Clone>(&self, inputs: &[T], mut gate: impl FnMut(&T, &T) -> T) -> T {
        assert_eq!(inputs.len(), self.input_count, "one value per input");
        let mut wires = inputs.to_vec();
        for [x, y] in &self.gates {
            let value = gate(&wires[*x], &wires[*y]);
            wires.push(value);
        }
        wires.swap_remove(self.output)
    }
}

/// The depth of the circuits a scheme evaluates.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Depth {
    /// The circuits of at most this depth: a leveled scheme.
    Bounded(usize),
    /// Every circuit: a scheme that bootstraps.
    Unbounded,
}

impl Depth {
    /// Whether the scheme evaluates the circuits of depth `depth`.
    pub fn admits(self, depth: usize) -> bool {
        match self {
            Self::Bounded(bound) => depth <= bound,
            Self::Unbounded => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Circuit, Depth};

    #[test]
    fn depth_is_the_longest_gate_path() {
        // x₀ · (x₁ · x₂) has depth 2; an input as the output has depth 0.
        assert_eq!(Circuit::new(3, vec![[1, 2], [0, 3]], 4).unwrap().depth(), 2);
        assert_eq!(Circuit::new(3, vec![[1, 2], [0, 3]], 3).unwrap().depth(), 1);
        assert_eq!(Circuit::new(2, vec![[0, 1]], 0).unwrap().depth(), 0);
        assert!(Circuit::new(2, vec![[0, 2]], 2).is_none());
        assert!(Circuit::new(2, vec![[0, 1]], 3).is_none());
    }

    #[test]
    fn bounded_depth_admits_shallower_circuits() {
        assert!(Depth::Bounded(2).admits(2));
        assert!(!Depth::Bounded(2).admits(3));
        assert!(Depth::Unbounded.admits(usize::MAX));
    }
}
