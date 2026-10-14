# AMD GPU対応の実装計画

この計画は、mxxの既存GPU実行経路をLinux上のAMD GPUへ移植し、NVIDIA CUDA対応も維持するためのものです。共通の演算ソースをCUDAとHIPでコンパイルし、Rustの`GpuRuntime::plan`から`execute`、成果物の保存と再読込まで接続します。HIPはAMDのGPUプログラミング環境ROCmのC++ APIです。

今回の成果物は計画書とAstra mediumによる計画レビューです。ソース実装、ビルド、テスト、GPUの確保や課金を伴う実行はこの計画作成では行いません。日本語はユーザーの明示指定に従います。実装時のrustdoc、通常のリポジトリ文書、コミット、PRは既存規約どおり英語にします。

## 対象と完成条件

対象はCargoワークスペース内のGPU機能です。`mxx-backends`の多項式、行列、NTT、RNSとCRT変換、整数と実数の制御演算、乱数と各サンプラ、trapdoorとpreimage、GPUグラフ実行とartifact I/Oを含めます。上位のDSL、KHE、BGVから既存APIで利用できることを完成条件とします。ユーザーの追加指示により、TFHEのAMD対応とblind rotation専用カーネルのHIP移植は後続段階へ移し、今回の完成条件から外します。CUDAのTFHEは維持します。

初期サポートはROCm公式対応のLinux環境とAMD GPUです。最初に検証するROCmの正確な版、OS、コンパイラ、GPU型番、`gfx`ターゲットは実装の最初の段階で固定します。RDNAのwave32機とCDNAのwave64機を別々に検証し、検証済み構成だけをサポート表へ記載します。AMD全製品、Windows、macOS、同一プロセス内のAMDとNVIDIA混在、ベンダ間でのGPU常駐ポインタ共有は対象外です。

ワークスペースから除外されている`crates/we`は今回の完成判定に含めません。外部CUDAライブラリ比較用の`crates/fhe/scripts/phantom_gpu_bgv.cu`、FIDESlib用のルート`CMakeLists.txt`と`third_party`はmxx本体のビルド経路から独立しており、AMD移植の対象外です。READMEにこの境界を明記し、AMDビルドがこれらを必要としないことを確認します。

完成には次の証拠が必要です。

- CUDA SDKのないAMD環境でHIP版をビルドし、既存と同じ`GpuRuntime`経路で演算、分岐、逐次ループ、preimage再試行、artifact I/OとBGVを実行できる。
- NVIDIAでは既存の`--features gpu`ビルドとGPU動作を維持し、CPUのみの構成ではGPU SDKを要求しない。
- 整数と剰余演算、NTT表現、符号と丸め、決定的ハッシュとシリアライズの意味を変えない。暗号パラメータ、棄却条件、誤差境界を移植の都合で変更しない。
- 常駐値の所有権、イベント依存、グラフ再束縛、保持中の出力、失敗時の出力抑止とplan poisoningを維持する。
- VRAMとpinned RAMは設定したwave幅とタイル幅に応じて増え、総反復数や総slot数に比例する保持を追加しない。演算をCPUへ移さない。
- 複数AMD GPUの物理構成を実機検証し、peer copyとhost stagingの双方を確認する。単一物理GPUでの論理複製だけで複数物理GPU対応を宣言しない。

## 現行コードから確認できる移植箇所

| 所有箇所 | 現状 | 必要な変更 |
| --- | --- | --- |
| `crates/backends/build.rs` | `gpu`でnvcc、`sm_89`既定、`cudart`と`cudadevrt`を使用。native source hashもCUDA固定 | HIPビルド分岐、適切な再ビルド条件、ベンダとコンパイラを含む識別 |
| `crates/fhe/build.rs` | `DEP_MXX_BACKENDS_CUDA_INCLUDE`を受け取り専用CUDAカーネルをビルド | 共通includeとバックエンド情報の受渡し。CUDAのみTFHE nativeをビルドし、HIPではBGVを独立してリンク |
| `crates/backends/cuda/src/Runtime.cu` | メモリpool、event、explicit graph、条件付きIFとWHILE、export slot、peer copy | 共通API層とベンダ別graph制御、HIPの所有権と可視性の検証 |
| `crates/backends/cuda/src/matrix/Matrix.cu` | 他のmatrix `.cu`と`ChaCha.cu`をincludeして構成 | HIPでも同じ翻訳単位を構築し、重複コンパイルを避ける |
| `matrix/MatrixNTT.cu`と`Control.cu` | 32 laneのshuffle、32 bit mask、warp内共有メモリ同期 | 論理32 laneの演算と物理wave幅を区別して移植 |
| `crates/fhe/cuda/tfhe_blind_rotation.cu` | cooperative grid同期、warp同期、PTX L2 prefetch、CUDA occupancy判定 | CUDAビルドのみ維持。AMD移植は後続段階 |
| `poly/dcrt/gpu.rs`、`backend/poly_gpu/fleet.rs` | opaque C ABIだがエラーと機器識別にCUDA名称とcompute capability | ベンダ、AMD gfx、wave幅、SDK情報を保持した共通機器識別 |
| `gpu_physical_control.rs`、`gpu_physical_lowering.rs`、`gpu_runtime_direct.rs` | `BranchIf`と`LoopWhile`をCUDA conditional bodyとしてコンパイル。既存のhost-driven loopもある | HIPでの分岐と反復を最後まで実行できる制御経路 |
| `gpu_execution_plan.rs`、`gpu_subgraph_kernel.rs` | plan contractと専用カーネルregistry | backend能力、native revision、専用カーネルの識別と照合 |
| `scripts/lib/repo_validation.py` | CUDAディレクトリと`*gpu*.rs`をGPU変更として検出 | 移動後のGPUソース、HIPビルドと制御変更の検出 |

RustのGPU実装は既に`gpu`を含むファイルに集約されています。新しいGPU専用Rustの型、制御、テストもこの規則に従います。具体的な算術とruntimeは引き続き`mxx-backends`が所有し、アプリケーションクレート同士の依存を追加しません。

## 採用する構成

### ビルド時のバックエンド選択

既存のCargo feature `gpu`は共通GPU機能を意味するものとして維持します。ビルド時環境変数`MXX_GPU_BACKEND=cuda|hip`を追加し、未指定は従来どおり`cuda`にします。`--features gpu`と`MXX_GPU_BACKEND=hip`がAMDの入口です。実行時のGPU自動検出でコンパイラを選ばず、1バイナリは1バックエンドだけをリンクします。

この方式なら全クレートへ排他的featureを伝播する変更やCargoのfeature加算によるCUDA/HIP同時有効化を避けられます。`gpu`有効時は不明な選択値をビルドエラーにし、HIP SDK不足やCUDA SDK不足を明確に報告します。`gpu`無効時はGPU SDK、GPU情報取得、GPUのlink処理を行いません。

CUDAの`CUDA_HOME`、`CUDA_LIB_DIR`、`NVCC`、`CUDA_ARCH`は維持します。HIPは`ROCM_PATH`、`HIPCC`を受け付け、`HIP_ARCH`に対象の`gfx`を明示します。コンパイラ、include、libパスの優先順をbuild scriptで統一し、HIP_PLATFORMがAMDであることを確認します。cross compile時に実機自動検出を必須にしません。各変数を`cargo::rerun-if-env-changed`へ登録し、ソース、header、backend、ターゲット、compiler版、ビルドフラグをnative revisionへ含めます。CUDAとHIPの成果物を識別し、切替で古いライブラリを再使用しないことを検証します。

共通nativeソースは`crates/backends/gpu/`へ移します。TFHEは後続段階まで`crates/fhe/cuda/`に残します。共通kernelは`.cu`のままCUDAでコンパイルし、HIPでは明示的にHIP言語としてコンパイルします。headerは`.h`へ整理します。`cc::Build::cuda(true)`はCUDA分岐だけで使い、HIPは`hipcc -x hip -std=c++17 -fPIC --offload-arch=<gfx>`でhost/deviceを含むobjectを生成し、archiveと`amdhip64`を適切にlinkします。追加のdevice linkが必要かは最初の小さなABIプローブで確定します。

`links = "mxx_backends"`を維持し、`gpu_include`、`gpu_backend`、対象arch、compilerとSDK情報をCargo metadataとして`mxx-fhe`へ渡します。TFHEのCUDAビルドは親と同じ選択とターゲットを使い、独立にnvccを選択しません。HIPでは`mxx-fhe/build.rs`がTFHE nativeのコンパイルをスキップします。同じmetadataからRustへ`cargo::rustc-check-cfg`とbackendのcfgを出し、native entryへのFFI参照もCUDA時だけにします。既存`gpu_blind_rotation_kernel`のOption APIではHIP時に`None`を返し、専用カーネル未提供をrustdocとサポート表に明記します。HIPのBGV buildでnvccや未定義TFHE symbolを要求しないことを確認します。TFHEのCPU APIは維持し、HIPでのTFHE利用は今回のサポート対象外です。旧CUDA専用include metadataは呼び出し側と同時に置き換え、二重インターフェースを残しません。

### 共通ソースとベンダ依存の境界

共有のNTT、matrix、CRT、samplerアルゴリズムを複製しません。HIPIFYは初期変換と未対応APIの棚卸しに使い、ビルドのたびに別実装を生成する方式は採用しません。

新設の`gpu/include/GpuPlatform.h`がstream、event、error、device props、allocation、copy、kernel launchと必要なintrinsicの共通名を定義します。共有コードではその名前を使い、CUDA/HIPの処理差は小さいベンダ実装へ限定します。CUDA名をHIP型へ大量にaliasする恒久的shimは作りません。条件付きgraphのCUDA専用処理はCUDAコンパイル専用のsourceへ分離し、HIPコンパイラにCUDA conditional型を見せません。

Rustとnative間は既存の`gpu_*`、`mxx_gpu_*`のopaque handleと値のABIを中心に維持します。公開subgraph headerは単独でincludeでき、上位crateがRuntimeやmatrixのprivate headerに依存しない形を保ちます。構造体のサイズ、alignment、patch offset、整数encodingを両コンパイラで確認します。

## HIPでの分岐と再試行

AMDの公式HIPIFY対応表では`cudaGraphConditionalHandleCreate`の対応HIP APIが空欄です。このため、単純なAPI置換で分岐やpreimage再試行が移植できるという前提は置きません。実装開始時の固定SDKで状況を再確認しますが、計画の基本方式はHIP graph regionを既存runtimeのhost制御から選択・反復する方法です。

これには小さい制御値のD2H転送と待ちが必要です。算術、比較、predicate生成、乱数、棄却判定、loop index更新はGPUで実行し、ホストは完了した制御値から次の事前コンパイル済みregionを選んで投入します。GPU演算をCPUでやり直す経路は追加しません。反復上限まで両branchやloop bodyを全部複製する方式は、グラフのサイズと無駄な計算が反復数に比例するため採用しません。

`GPU.md`の非同期wrapper規則に従い、D2Hを行わないnative/Rust wrapperはeventを返して直ちに戻ります。必要な待ちは、制御値をD2Hで取得するruntimeの制御境界に限り、対象streamの完了eventで行います。device全体同期、計算wrapper内のstream同期、graph host callback内でのHIP API呼出しは使いません。複数独立laneの制御recordはまとめて転送し、待ちの必要なlane以外の演算を止めない設計にします。

### 制御経路の実装手順

1. `gpu_execution_plan.rs`へnative conditionalとhost-scheduled regionの区別を追加する。CUDAは従来のconditional nodeを維持し、HIPはcontrol boundaryをもつregionへloweringする。単なる`Unsupported`による機能削減を完成と扱わない。
2. `gpu_physical_control.rs`と`gpu_physical_lowering.rs`でIFとWHILE、preimage retryのbodyを再利用可能なregionにする。親bodyと子bodyのbinding、loop-carried value、最終output、status、scratch、artifact occurrenceを明示する。入れ子の制御も同じ方式で扱う。
3. IFはpredicate生成regionの終了後に制御recordを読み、選んだbodyだけを実行する。非選択branchのstatus、出力、乱数、artifact publicationは実行しない。合流後のbindingは選択結果へ結ぶ。
4. WHILEはloop index、要求回数、statusをGPU上で保持し、事前に確定した上限を超える要求は同じDeviceStatusとして扱う。0回、最終反復、途中失敗、入れ子、loop-carried値の更新順を維持する。
5. preimage retryはGPUでacceptを判定し、accept済み列を上書きしない。attemptに応じたseed導出、`max_attempts`、attempts/accepted/error_codeの意味、hard cutoff、全列未accept時の出力抑止を維持する。bodyとworkspaceは再利用する。
6. `gpu_runtime_direct.rs`の通常実行、planning trial、I/O trial、node profile、`predicted_seconds`へ同じ制御スケジュールを接続する。性能測定でD2Hとevent待ちと再投入費用を除外しない。planの入力再束縛では制御bufferとbody operandも更新する。
7. region ownerとeventを実際の最終consumerまで保持する。取消・失敗・launch uncertaintyではqueued workが完了する前にplanやbufferを解放しない。既存のpoisoningとfail-closedの解放方針を維持する。

HIPで増える制御転送と起動回数は実測してreportへ出します。これは性能上の主要リスクです。重い逐次loopで著しい低下がある場合は、その生産経路のGPU専用subgraph kernelで融合する別の実装単位を切り出します。最初から全kernelをpersistent dispatcherへ再設計することはしません。

## 演算カーネルとメモリの移植

### wave幅と整数演算

AMDには32 laneと64 laneのwaveがあるため、物理waveを32と決め打ちしません。一方、既存NTTとinteger matvecの論理32 laneタイルは意味を維持し、HIP側のshuffleではwidth=32と64 bitの正しいactive maskを使います。wave64の上半分でも別の32 laneタイルとして結果を得ることを検証します。`lane`、row割当、blockサイズ、部分tile、radix、NTT boundary twistとbit reversalをまとめて監査します。

NTTの共有メモリ同期では、CUDAのwarp内同期省略がAMDでも成立するとは仮定しません。タイル内同期を正しく定義できない箇所は一様なblock barrierへ変更し、全threadが到達することを確認します。warp同期を単なるno-opにしません。変更後にNTTの性能と正しさを確認します。

`__umul64hi`、ShoupとBarrett、u32/u64算術、hostの`unsigned __int128`、符号拡張、overflow判定、alignment、FP64、math関数を監査します。整数の結果は信頼済みCPU/OpenFHEと完全一致を要求します。浮動小数の内部値は既存の許容誤差とnorm契約で判定し、ベンダ間の全bit一致を要求して棄却条件や分布を変えないようにします。ChaChaとhash samplingは同じ入力鍵とtagに対してbyte一致を要求します。GPUの乱数初期化をCPU乱数に置き換えません。

### 非同期所有権と成果物の公開

HIPでもstream ordered allocation/free、async copy、producer eventとconsumer stream wait、pinned bufferのevent後解放を使います。`hipMallocAsync`などが存在するだけで対応完了とせず、固定SDKと実機でpool属性、graphからのアクセス、再束縛、OOMの型付き分類を確かめます。実行中に同期allocationへ切り替える回避策は追加しません。

export slotの`__threadfence_system`、ready store、ホストacquire readはROCmのcoherent pinned memoryとsystem visibilityを検証します。CUDAのPCIe公開手順をそのまま安全と仮定しません。payloadとheaderの順序を実機で保証できない場合は、D2H payloadとheader copy、完了eventを明示的に用いる方式へ統一し、I/O workerはevent完了後にだけpayloadを読むよう接続します。未完了readやready pollingでCPUを占有する経路を追加しません。保存形式は維持し、CUDAで保存したcanonical artifactをHIPで読めること、逆方向も確認します。

複数GPUでは`detected_gpu_device_ids`と`mxx_set_device`、論理mapping、matrix全limbを同一deviceに置く方針を維持します。peer accessとpool accessを方向別に確認し、使えなければ既存同様にbounded pinned stagingとasync D2H/H2D、event依存でcopy nodeを実現します。HIPのgraphに異なるdeviceのnodeを入れられると仮定せず、必要ならdeviceごとのregionとeventで接続します。`MXX_GPU_HOST_STAGED_COPIES=1`は引き続き検証に使えます。

### 後続段階のTFHE専用カーネル

この節は今回の実装と完成条件に含めません。初期AMD対応のaccept後に追加レビューする作業です。

`TfheParams::gpu_blind_rotation_kernel`とregistryのproduction登録経路を維持します。PTX `prefetch.global.L2`はCUDA側だけにし、HIPでは機能上不要なhintとして省略するか対応intrinsicを実測後に用います。

HIPのcooperative launch、grid barrier、dynamic shared memory、occupancyとGPU属性を確認し、grid全体が同時常駐できるときに協調カーネルを使います。不可の場合はblind rotationの同じGPU subgraphを段階kernelとregionに分け、grid barrierに対応するkernel境界で順序付けします。CPUへ移したり専用subgraphを黙って無効化したりしません。native登録前に能力とparameter制約を確認し、選択したstrategyをplanとnative revisionに記録します。bit単位比較には既存のDSL bodyを信頼済みreferenceとして使います。

## 機器識別とplanの契約

`GpuDeviceIdentity`はbackend、GPU名、UUID等の安定識別子、CUDA SMまたはAMD gfx、wave幅、VRAM、driver/runtime版を持つよう変更します。AMD gfxをCUDAのcompute_major/minorへ偽装しません。C ABIとRustの定義を同時に更新します。

`runtime_backend_identity`の`cuda-fleet`固定を外し、ベンダ、機器、実行owner、能力、native revisionが区別できるidentityにします。plan contractと測定再利用のkeyにはbackend、arch、SDKとcompiler版、ソースとビルドフラグ、subgraph kernel revisionを含めます。異なるbackendや変更前nativeへのplan再利用を拒否します。共通canonical artifactとGPU固有planを混同せず、保存形式とprotocol claimは変更しません。

## 実装の順番と各段階の判定

| 段階 | 接続する変更 | 段階を終える証拠 |
| --- | --- | --- |
| 1 | 固定SDKと機器の選定、全CUDA API/intrinsic棚卸し、HIP buildとC ABIの小さい実験 | async allocator、graph patch/update、export visibility、cross-device copyの可否表と選択strategy。未対応項目を隠さない |
| 2 | featureは維持してbuild selector、共通source移動、platform API、include metadata、FHEのCUDA専用TFHE分離を一括変更 | CPU、CUDA、HIPのlib compileと切替再ビルド。HIP branchからCUDA header/linkが消える |
| 3 | NTT、matrix、RNS/CRT、control、hash、sampler、所有権、I/O、機器識別をproduction pathに接続 | 信頼済みCPUとの差分unit検証、非同期寿命、再束縛、artifact round-trip |
| 4 | HIP制御region、入れ子IF/WHILE、preimage retry、trialとprofile、multi-device regionを一括接続 | 同じ公開runtimeで0回/複数回/失敗/再実行を検証し、正しいstatusと所有権が維持される |
| 5 | 両wave系統、NVIDIA回帰、複数物理AMD GPU、メモリと性能、runnerと文書 | 下記検証表を満たす。サポート表と制限を実測に合わせて確定 |

段階1の実験はAPIや同期の不確実性を解くための狭いチェックです。その後は各段階の型、API、native、caller、テストを一貫した変更単位として実装してから検証します。変更のない成功済みチェックを繰り返しません。GPUが確保できなければcompile evidenceと実機未検証を区別し、AMD対応完成とは宣言しません。

## 検証の内容

この表は将来の実装検証手順であり、計画レビュー中には実行しません。integration testは現在の依頼では許可されていないため実行せず、将来の実装依頼で明示された場合だけ実行します。必要なproductionグラフ検証は`*gpu*.rs`内のunit testでも行えるようにします。

| 検証対象 | 必須ケースと判定 |
| --- | --- |
| ビルド | CPU、既定CUDA、明示CUDA、明示HIP。CUDAなしAMD環境、不明backend、SDK欠落、arch不正、backend切替、header更新とTFHEへのmetadata伝播 |
| 算術 | add/sub/mul、small RHS、NTT forward/inverseとformat、CRT/RNS、mod switch、decompose、pack/codec、integer controlをCPU/OpenFHEと比較。1 limb/複数limb、strided view、tail、複数NTT radix、既存上限131072を含む対応degree |
| 乱数とpreimage | hashとseed導出の一致、既存samplerのnormと分布、初回accept、複数retry、上限exhaust、再束縛時のfresh randomness。確率的norm失敗は統計的に判定 |
| 制御 | IF両側、非選択側の失敗とartifact非公開、0/1/複数loop、上限超過、入れ子、loop-carried matrix、retryとbranch内のparallel wave、status後の再実行 |
| 所有権 | 入力を別planから供給、出力を保持したまま次回実行、異なるstreamのproducer、早期drop、OOM、部分投入とlaunch uncertainty、解放worker、stale plan拒否 |
| artifact | Memory/File/Device storeの保存と読込、selected family、chunked export、pinned slot可視性、試行のdiscard、CUDAとHIPのcanonical互換 |
| FHE | BGV全levelとmultilimb、keygenから復号までのproduction graph。CUDA TFHEは既存回帰を維持し、HIPで専用kernel未提供でもBGVがリンクして動くことを確認。TFHEのAMD実機検証は後続 |
| 複数GPU | mapping未設定、`0,0`、`0,0,0`、複数物理AMD、peer可/不可、強制host staging、全部limbの同一device、各deviceのVRAM予算 |
| 非同期と性能 | wrapperがD2H以外で待たないことをtimelineとsourceで確認。起動回数、転送byteと回数、制御待ち、latency、total_time、max_parallelism、VRAM/pinned RAM peakを同じproduction pathで記録 |

実装時の基本compile gateは次のとおりです。HIP_ARCHの`<gfx>`は固定した検証機に合わせて実値へ置換します。各backendの成果物とログを分離します。

```bash
cargo +nightly fmt --all
cargo test -r --workspace --lib --no-run
MXX_GPU_BACKEND=cuda cargo test -r --workspace --lib --features gpu --no-run
MXX_GPU_BACKEND=hip HIP_ARCH=<gfx> cargo test -r --workspace --lib --features gpu --no-run
```

warningのないビルドを要求します。次に、ビルドしたunit test binaryをexact filterと同じparameterで実行します。GPU依存testは`#[ignore = "requires a supported GPU"]`へ統一し、`--ignored`で選択実行します。既存testの判定を弱めず、random seedを固定しません。hash一致比較はその実行でランダムに生成した同じkeyを両実装へ渡します。

同期バグを扱う検証は300回、round-trip smokeは3〜5回を`GPU.md`に従って行い、失敗が出ても許可と資源の範囲で完了数と失敗数を収集します。同期と所有権の移植は少なくとも影響する300回のセットをCUDAとHIPで設けます。multi-device変更はidentity、`0,0`、`0,0,0`の各modeを確認します。既存codeに論理複製制約があれば実装初期に把握し、テストのためだけに別経路を作らずfleetとcontractの表現を揃えます。

パラメータは環境変数で小さい既定値から変更可能にします。各testは専用`test_data`ディレクトリを使い、ユーザーの既存artifactは保存します。実GPU testは承認済みのsandbox外実行機構を使用し、拒否を迂回しません。今回の計画作成は新たなクラウド契約、pod作成、remote実行を許可するものではありません。

性能は変更前CUDAと変更後CUDAを同じNVIDIA機、同じgraphとparameterで比較します。AMDとNVIDIAの結果は別機種であることを併記します。新HIPには変更前AMD値がないため、最初の正しいHIP実行を以降の最適化のbaselineにします。大きな退行を観測したらstage別に原因を調べ、測定なしに同等性能を約束しません。総slot数と反復数だけを増やしたケースでpeak memoryが増え続けないことを確かめます。計測は既存のlatencyとtotal_timeの定義を守ります。

## CIとドキュメント

`scripts/run_tests.sh`と`repo_validation.py`をbackend selectorに対応させ、通常CPU CIを維持しながらCUDA/HIPそれぞれのwarning-free lib compileを追加します。GPU実行には実機runnerを用い、存在しないAMD runnerを仮定しません。runner準備までの手動検証記録も固定環境、source identity、command、環境変数、終了statusを保存します。integration testを通常CIに勝手に追加しません。

`README.md`、`crates/backends/README.md`と`SPEC.md`、`crates/fhe/README.md`、変更したGPU moduleのrustdoc、`GPU.md`と`BUILDER.md`を実装と同時に更新します。CUDA専用の名称、header path、ignore理由を更新し、backend選択、HIPの必要条件、検証済みサポート表、HIPの制御転送費用、multi-deviceの制約、外部CUDA比較ツールの範囲を記載します。runtimeの画面やHTML reportにはGPU利用者が判断に必要なbackendと機器と計測を表示します。

## 外部資料と未検証事項

調査日現在の公式資料を参照していますが、`latest`は動的です。実装段階1では実際のSDK版に対応する資料とheader、実機結果で判断を確定します。

- [AMD HIP移植ガイド](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html)：HIP compiler、warp幅とlane maskの移植事項。
- [AMD HIP graphの説明](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/hipgraph.html)：graph作成と非同期メモリAPI。
- [AMD HIPIFYのCUDA Runtime API対応表](https://rocm.docs.amd.com/projects/HIPIFY/en/latest/reference/tables/CUDA_Runtime_API_functions_supported_by_HIP.html)：conditional handleの対応欄から、単純移植できないと判断した根拠。
- [AMD cooperative groups](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/cooperative_groups.html)：grid group、cooperative launchとAMDのthread scheduling。
- [AMD ROCm互換表](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)：検証機とOS/SDKの選定。

実GPUのAPI、同期、算術、速度はこの計画作成では未検証です。計画のacceptは設計と実装手順への評価であり、AMD対応の実装完了や実機保証を意味しません。
