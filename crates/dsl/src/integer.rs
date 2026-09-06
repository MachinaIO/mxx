use super::*;
use mxx_ir_core::node::IntBinaryOp;

#[derive(Clone)]
pub struct Int {
    pub(super) value: ValueHandle,
    pub(super) pending: Pending,
}

#[derive(Clone)]
pub struct Bool {
    pub(super) value: ValueHandle,
    pub(super) pending: Pending,
}

impl Int {
    pub fn constant(value: impl Into<num_bigint::BigInt>) -> Self {
        let node = NodeHandle::new(
            NodeKind::ConstantInt(value.into()),
            Vec::new(),
            vec![WireType::ConstantInt],
        );
        Self { value: node.output(0).expect("constant integer"), pending: Pending::default() }
    }

    pub fn evaluate(expression: impl Into<IntExpr>) -> Self {
        let node = NodeHandle::new(
            NodeKind::EvaluateInt(expression.into()),
            Vec::new(),
            vec![WireType::ConstantInt],
        );
        Self { value: node.output(0).expect("evaluated integer"), pending: Pending::default() }
    }

    pub fn value_handle(&self) -> &ValueHandle {
        &self.value
    }

    pub fn pending_assumptions(&self) -> bool {
        false
    }

    pub fn add(self, rhs: impl Into<Int>) -> Self {
        self.binary(rhs.into(), mxx_ir_core::node::IntBinaryOp::Add, "integer sum")
    }

    pub fn sub(self, rhs: impl Into<Int>) -> Self {
        self.binary(rhs.into(), mxx_ir_core::node::IntBinaryOp::Subtract, "integer difference")
    }

    pub fn mul(self, rhs: impl Into<Int>) -> Self {
        self.binary(rhs.into(), mxx_ir_core::node::IntBinaryOp::Multiply, "integer product")
    }

    /// Divides by the absolute divisor and rounds down; a zero divisor fails at runtime.
    pub fn div(self, rhs: impl Into<Int>) -> Self {
        self.binary(rhs.into(), mxx_ir_core::node::IntBinaryOp::Divide, "integer quotient")
    }

    /// Returns the nonnegative remainder modulo the absolute divisor.
    pub fn rem(self, rhs: impl Into<Int>) -> Self {
        self.binary(rhs.into(), mxx_ir_core::node::IntBinaryOp::Remainder, "integer remainder")
    }

    fn binary(
        self,
        rhs: Self,
        operation: mxx_ir_core::node::IntBinaryOp,
        output_name: &'static str,
    ) -> Self {
        let pending = Pending::merge([self.pending, rhs.pending]);
        let node = NodeHandle::new(
            NodeKind::IntBinary(operation),
            vec![self.value, rhs.value],
            vec![WireType::Int],
        );
        Self { value: node.output(0).expect(output_name), pending }
    }

    pub fn equal(self, rhs: impl Into<Int>) -> Bool {
        self.compare(rhs.into(), mxx_ir_core::node::IntCompareOp::Equal)
    }

    pub fn less_equal(self, rhs: impl Into<Int>) -> Bool {
        self.compare(rhs.into(), mxx_ir_core::node::IntCompareOp::LessEqual)
    }

    pub fn less(self, rhs: impl Into<Int>) -> Bool {
        self.compare(rhs.into(), mxx_ir_core::node::IntCompareOp::Less)
    }

    pub fn bit(self, position: impl Into<Int>) -> Result<Bool, DslError> {
        let position = position.into();
        let bit = position.compile_expression().ok_or(DslError::CompileTimeIndex)?;
        let node =
            NodeHandle::new(NodeKind::BitExtract { bit }, vec![self.value], vec![WireType::Bool]);
        Ok(Bool {
            value: node.output(0).expect("integer bit"),
            pending: Pending::merge([self.pending, position.pending]),
        })
    }

    #[track_caller]
    pub fn lift_to_constant_polynomial(self, matrix_type: MatrixType) -> Mat {
        assert_eq!(matrix_type.rows, IntExpr::constant(1), "constant-polynomial lift is scalar");
        assert_eq!(matrix_type.columns, IntExpr::constant(1), "constant-polynomial lift is scalar");
        let pending = self.pending;
        let node = NodeHandle::new(
            NodeKind::LiftIntegerToConstantPolynomial { matrix_type: matrix_type.clone() },
            vec![self.value],
            vec![WireType::Matrix(matrix_type.clone())],
        );
        Mat { value: node.output(0).expect("constant-polynomial lift"), matrix_type, pending }
    }

    fn compare(self, rhs: Self, operation: mxx_ir_core::node::IntCompareOp) -> Bool {
        let pending = Pending::merge([self.pending, rhs.pending]);
        let node = NodeHandle::new(
            NodeKind::IntCompare(operation),
            vec![self.value, rhs.value],
            vec![WireType::Bool],
        );
        Bool { value: node.output(0).expect("integer comparison"), pending }
    }
}

impl Bool {
    #[doc(hidden)]
    pub fn value_handle(&self) -> &ValueHandle {
        &self.value
    }

    pub fn constant(value: bool) -> Self {
        let node = NodeHandle::new(
            NodeKind::ConstantBool(value),
            Vec::new(),
            vec![WireType::ConstantBool],
        );
        Self { value: node.output(0).expect("constant boolean"), pending: Pending::default() }
    }

    pub fn to_int(self) -> Int {
        let node = NodeHandle::new(NodeKind::BoolToInt, vec![self.value], vec![WireType::Int]);
        Int { value: node.output(0).expect("boolean integer"), pending: self.pending }
    }
}

#[derive(Clone)]
pub struct Bytes {
    pub(super) value: ValueHandle,
    pub(super) pending: Pending,
}

impl Bytes {
    #[doc(hidden)]
    pub fn value_handle(&self) -> &ValueHandle {
        &self.value
    }
}

impl Int {
    /// Returns the expression for metadata that must be resolved before execution.
    /// Runtime input integers cannot be used as matrix shapes or static slice bounds.
    pub fn expression(&self) -> Result<IntExpr, DslError> {
        self.compile_expression().ok_or(DslError::CompileTimeIndex)
    }

    pub(super) fn compile_expression(&self) -> Option<IntExpr> {
        fn expression(value: &ValueHandle) -> Option<IntExpr> {
            if !value
                .construction_scope()
                .is_ancestor_of(&mxx_ir_core::current_construction_scope())
            {
                return None;
            }
            match value.node().kind() {
                NodeKind::ConstantInt(value) => Some(IntExpr::constant(value.clone())),
                NodeKind::EvaluateInt(value) => Some(value.clone()),
                NodeKind::IntBinary(op) => {
                    let args = value.node().arguments();
                    let left = Box::new(expression(&args[0])?);
                    let right = Box::new(expression(&args[1])?);
                    Some(
                        match op {
                            IntBinaryOp::Add => IntExpr::Add(left, right),
                            IntBinaryOp::Subtract => IntExpr::Sub(left, right),
                            IntBinaryOp::Multiply => IntExpr::Mul(left, right),
                            IntBinaryOp::Divide | IntBinaryOp::Remainder => {
                                // Runtime integers use Euclidean division. Floor division agrees
                                // only for a positive divisor; exact IntExpr::Div does not agree.
                                let IntExpr::Const(divisor) = right.as_ref() else { return None };
                                if divisor <= &num_bigint::BigInt::from(0) {
                                    return None;
                                }
                                if matches!(op, IntBinaryOp::Divide) {
                                    IntExpr::FloorDiv(left, right)
                                } else {
                                    IntExpr::Rem(left, right)
                                }
                            }
                        }
                        .canonicalize(),
                    )
                }
                _ => None,
            }
        }
        expression(&self.value)
    }
}

pub(super) fn has_loop_index(value: &IntExpr) -> bool {
    match value {
        IntExpr::LoopIndex(_) => true,
        IntExpr::Const(_) | IntExpr::Var(_) => false,
        IntExpr::Add(a, b) |
        IntExpr::Sub(a, b) |
        IntExpr::Mul(a, b) |
        IntExpr::Div(a, b) |
        IntExpr::FloorDiv(a, b) |
        IntExpr::Rem(a, b) |
        IntExpr::RoundDiv(a, b) => has_loop_index(a) || has_loop_index(b),
        IntExpr::Log2Ceil(value) => has_loop_index(value),
        IntExpr::Select { selector, branches } => {
            has_loop_index(selector) || branches.iter().any(has_loop_index)
        }
    }
}

impl From<IntExpr> for Int {
    fn from(value: IntExpr) -> Self {
        Self::evaluate(value)
    }
}
impl From<&IntExpr> for Int {
    fn from(value: &IntExpr) -> Self {
        Self::evaluate(value.clone())
    }
}
impl From<&Int> for Int {
    fn from(value: &Int) -> Self {
        value.clone()
    }
}
impl From<Bool> for Int {
    fn from(value: Bool) -> Self {
        value.to_int()
    }
}
impl From<&Bool> for Int {
    fn from(value: &Bool) -> Self {
        value.clone().to_int()
    }
}

macro_rules! integer_constants {
    ($($ty:ty),*) => { $(impl From<$ty> for Int {
        fn from(value: $ty) -> Self { Self::constant(value) }
    })* };
}
integer_constants!(
    i8,
    i16,
    i32,
    i64,
    i128,
    isize,
    u8,
    u16,
    u32,
    u64,
    u128,
    usize,
    num_bigint::BigInt,
    num_bigint::BigUint
);

impl From<&num_bigint::BigInt> for Int {
    fn from(value: &num_bigint::BigInt) -> Self {
        Self::constant(value.clone())
    }
}
impl From<&num_bigint::BigUint> for Int {
    fn from(value: &num_bigint::BigUint) -> Self {
        Self::constant(value.clone())
    }
}

macro_rules! integer_operator {
    ($trait:ident, $method:ident) => {
        impl<R: Into<Int>> std::ops::$trait<R> for Int {
            type Output = Int;
            fn $method(self, rhs: R) -> Int {
                Int::$method(self, rhs)
            }
        }
        impl<R: Into<Int>> std::ops::$trait<R> for &Int {
            type Output = Int;
            fn $method(self, rhs: R) -> Int {
                Int::$method(self.clone(), rhs)
            }
        }
    };
}
integer_operator!(Add, add);
integer_operator!(Sub, sub);
integer_operator!(Mul, mul);
integer_operator!(Div, div);
integer_operator!(Rem, rem);

impl std::ops::Neg for Int {
    type Output = Int;
    fn neg(self) -> Int {
        Int::constant(0) - self
    }
}

impl std::ops::Neg for &Int {
    type Output = Int;
    fn neg(self) -> Int {
        -self.clone()
    }
}

macro_rules! integer_left_operator {
    ($trait:ident, $method:ident; $($ty:ty),* $(,)?) => { $(
        impl std::ops::$trait<Int> for $ty {
            type Output = Int;
            fn $method(self, rhs: Int) -> Int { Int::$method(Int::from(self), rhs) }
        }
        impl std::ops::$trait<&Int> for $ty {
            type Output = Int;
            fn $method(self, rhs: &Int) -> Int { Int::$method(Int::from(self), rhs) }
        }
    )* };
}
macro_rules! integer_left_operators {
    ($($ty:ty),* $(,)?) => {
        integer_left_operator!(Add, add; $($ty),*);
        integer_left_operator!(Sub, sub; $($ty),*);
        integer_left_operator!(Mul, mul; $($ty),*);
        integer_left_operator!(Div, div; $($ty),*);
        integer_left_operator!(Rem, rem; $($ty),*);
    };
}
integer_left_operators!(
    i32,
    num_bigint::BigInt,
    num_bigint::BigUint,
    &num_bigint::BigInt,
    &num_bigint::BigUint,
    IntExpr,
    &IntExpr
);
