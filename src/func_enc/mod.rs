pub mod aky24;

pub trait FuncEnc {
    type Params;
    type EncKey;
    type MasterKey;
    type Msg;
    type Ciphertext;
    type Func;
    type FuncKey;
    type Output;

    fn setup(&self, params: &Self::Params) -> (Self::EncKey, Self::MasterKey);

    fn enc(
        &self,
        params: &Self::Params,
        enc_key: &Self::EncKey,
        msg: &Self::Msg,
    ) -> Self::Ciphertext;

    fn keygen(
        &self,
        params: &Self::Params,
        msk: &Self::MasterKey,
        func: &Self::Func,
    ) -> Self::FuncKey;

    fn dec(
        &self,
        params: &Self::Params,
        ct: &Self::Ciphertext,
        fsk: &Self::FuncKey,
    ) -> Self::Output;
}
