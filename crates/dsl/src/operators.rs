//! Arithmetic notation for graph values. Operators construct the same primitive nodes as methods.

use super::*;
use std::ops::{BitAnd, BitOr, BitXor, Not};

macro_rules! borrowed_matrix_operator {
    ($trait:ident, $method:ident) => {
        impl $trait<&Mat> for Mat {
            type Output = Mat;
            #[track_caller]
            fn $method(self, rhs: &Mat) -> Mat {
                $trait::$method(self, rhs.clone())
            }
        }
        impl $trait<Mat> for &Mat {
            type Output = Mat;
            #[track_caller]
            fn $method(self, rhs: Mat) -> Mat {
                $trait::$method(self.clone(), rhs)
            }
        }
        impl $trait<&Mat> for &Mat {
            type Output = Mat;
            #[track_caller]
            fn $method(self, rhs: &Mat) -> Mat {
                $trait::$method(self.clone(), rhs.clone())
            }
        }
    };
}
borrowed_matrix_operator!(Add, add);
borrowed_matrix_operator!(Sub, sub);
borrowed_matrix_operator!(Mul, mul);

impl Neg for &Mat {
    type Output = Mat;
    #[track_caller]
    fn neg(self) -> Mat {
        -self.clone()
    }
}

#[track_caller]
fn constant_scalar(matrix: &Mat, value: IntExpr) -> Mat {
    Ring::new(matrix.matrix_type.modulus.clone(), matrix.matrix_type.ring_dimension.clone())
        .polynomial([value])
}

impl<R: Into<IntExpr>> Mul<R> for Mat {
    type Output = Mat;
    #[track_caller]
    fn mul(self, rhs: R) -> Mat {
        let scalar = constant_scalar(&self, rhs.into());
        self * scalar
    }
}

impl<R: Into<IntExpr>> Mul<R> for &Mat {
    type Output = Mat;
    #[track_caller]
    fn mul(self, rhs: R) -> Mat {
        self.clone() * rhs
    }
}

impl Mul<Mat> for i32 {
    type Output = Mat;
    #[track_caller]
    fn mul(self, rhs: Mat) -> Mat {
        let scalar = constant_scalar(&rhs, IntExpr::from(self));
        scalar * rhs
    }
}

impl Mul<&Mat> for i32 {
    type Output = Mat;
    #[track_caller]
    fn mul(self, rhs: &Mat) -> Mat {
        self * rhs.clone()
    }
}

impl From<bool> for Bool {
    fn from(value: bool) -> Self {
        Self::constant(value)
    }
}
impl From<&bool> for Bool {
    fn from(value: &bool) -> Self {
        Self::constant(*value)
    }
}
impl From<&Bool> for Bool {
    fn from(value: &Bool) -> Self {
        value.clone()
    }
}

impl<R: Into<Bool>> BitAnd<R> for Bool {
    type Output = Bool;
    fn bitand(self, rhs: R) -> Bool {
        // The closed Boolean-interval decoder contract recognizes this exact
        // conjunction form, including both BoolToInt operands and the threshold two.
        (self.to_int() + rhs.into().to_int()).equal(2)
    }
}

impl<R: Into<Bool>> BitOr<R> for Bool {
    type Output = Bool;
    fn bitor(self, rhs: R) -> Bool {
        Int::constant(1).less_equal(self.to_int() + rhs.into().to_int())
    }
}

impl<R: Into<Bool>> BitXor<R> for Bool {
    type Output = Bool;
    fn bitxor(self, rhs: R) -> Bool {
        (self.to_int() + rhs.into().to_int()).equal(1)
    }
}

impl Not for Bool {
    type Output = Bool;
    fn not(self) -> Bool {
        self.to_int().equal(0)
    }
}

impl Not for &Bool {
    type Output = Bool;
    fn not(self) -> Bool {
        !self.clone()
    }
}

macro_rules! borrowed_boolean_operator {
    ($trait:ident, $method:ident) => {
        impl<R: Into<Bool>> $trait<R> for &Bool {
            type Output = Bool;
            fn $method(self, rhs: R) -> Bool {
                $trait::$method(self.clone(), rhs)
            }
        }
        impl $trait<Bool> for bool {
            type Output = Bool;
            fn $method(self, rhs: Bool) -> Bool {
                $trait::$method(Bool::from(self), rhs)
            }
        }
        impl $trait<&Bool> for bool {
            type Output = Bool;
            fn $method(self, rhs: &Bool) -> Bool {
                $trait::$method(Bool::from(self), rhs)
            }
        }
    };
}
borrowed_boolean_operator!(BitAnd, bitand);
borrowed_boolean_operator!(BitOr, bitor);
borrowed_boolean_operator!(BitXor, bitxor);

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;

    #[test]
    fn borrowed_matrix_arithmetic_retains_one_sampler() {
        let ring = Ring::new(257, 8);
        let sample = ring.gaussian((2, 3), 3, 19);
        let output = -&sample + &sample * 3 + 3 * &sample - sample.clone();
        assert_eq!(output.matrix_type(), sample.matrix_type());
        let built = DslContext::new("borrowed-matrix-arithmetic")
            .output("result", output)
            .unwrap()
            .build()
            .unwrap();
        built.validate(&ParamEnv::default()).unwrap();
        assert_eq!(
            built
                .graph
                .root_scope()
                .nodes()
                .iter()
                .filter(|node| matches!(node.kind(), NodeKind::GaussianSample { .. }))
                .count(),
            1
        );
    }

    #[test]
    fn integer_operators_preserve_big_integer_and_euclidean_division_semantics() {
        let magnitude = BigInt::from(1u8) << 200usize;
        let value = Int::from(&magnitude);
        assert_eq!(
            (-&value).expression().unwrap().evaluate(&ParamEnv::default()).unwrap(),
            -&magnitude
        );
        let expression = (7 - &value).add(&value).mul(3).sub(1).div(2).rem(4);
        assert_eq!(
            expression.expression().unwrap().evaluate(&ParamEnv::default()).unwrap(),
            BigInt::from(2)
        );
        assert_eq!(
            (Int::constant(-7) / 3).expression().unwrap().evaluate(&ParamEnv::default()).unwrap(),
            BigInt::from(-3)
        );
        assert_eq!(
            (Int::constant(-7) % 3).expression().unwrap().evaluate(&ParamEnv::default()).unwrap(),
            BigInt::from(2)
        );
        assert!((Int::constant(7) / -3).expression().is_err());
        let built = DslContext::new("integer-comparisons")
            .output("equal", value.clone().equal(&value))
            .unwrap()
            .output("less", value.clone().less(&magnitude + 1))
            .unwrap()
            .output("less-equal", value.less_equal(&magnitude))
            .unwrap()
            .build()
            .unwrap();
        built.validate(&ParamEnv::default()).unwrap();
    }

    #[test]
    fn boolean_conjunction_preserves_the_closed_decoder_form() {
        use mxx_ir_core::node::{IntBinaryOp, IntCompareOp};
        let ring = Ring::new(17, 1);
        let left = ring.bool_input("left");
        let right = ring.bool_input("right");
        let output = left & right;
        let equality = output.value_handle().node();
        assert!(matches!(equality.kind(), NodeKind::IntCompare(IntCompareOp::Equal)));
        let [sum, threshold] = equality.arguments() else { panic!("binary comparison") };
        assert!(
            matches!(threshold.node().kind(), NodeKind::ConstantInt(value) if value == &BigInt::from(2))
        );
        assert!(matches!(sum.node().kind(), NodeKind::IntBinary(IntBinaryOp::Add)));
        assert_eq!(sum.node().arguments().len(), 2);
        assert!(
            sum.node()
                .arguments()
                .iter()
                .all(|argument| matches!(argument.node().kind(), NodeKind::BoolToInt))
        );
    }

    #[test]
    fn boolean_operator_results_are_executable_boolean_wires() {
        let ring = Ring::new(17, 1);
        let left = ring.bool_input("left");
        let right = ring.bool_input("right");
        let outputs = (&left & &right, &left | false, true ^ &right, !&left);
        let built = DslContext::new("boolean-operators")
            .output("results", outputs)
            .unwrap()
            .build()
            .unwrap();
        let validated = built.validate(&ParamEnv::default()).unwrap();
        assert_eq!(built.graph.outputs().len(), 4);
        assert!(built.graph.outputs().values().all(|output| matches!(
            validated.root_scope().wire_types[&output.value],
            mxx_ir_core::types::ConcreteWireType::Bool
        )));
    }
}
