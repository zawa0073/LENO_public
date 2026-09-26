# LENO training: rectangular grids and disk nodes

矩形格子と円盤上の反応拡散データから、1ステップの時間発展を学習するコードです。
共通する処理には同じ名前を使い、領域・データ形状に依存する処理には `Disk` / `Nodes` を付けています。
実行入口は `squaredomain.py` と `circledomain.py` です。
モデルと基底処理はそれぞれ `model_squaredomain.py` と `model_circledomain.py` にまとめています。
**4つのPythonファイルを同じフォルダに置いてください。**

## ファイル構成と読む順序

| ファイル | 内容 |
|---|---|
| [squaredomain.py](squaredomain.py) | 正方形・矩形格子のデータ読み込み、学習、評価、描画、CLI |
| [model_squaredomain.py](model_squaredomain.py) | 格子の質量行列、Neumann・周期基底、スペクトル層、`LENO2d` |
| [circledomain.py](circledomain.py) | 円盤データの読み込み、学習、評価、描画、CLI |
| [model_circledomain.py](model_circledomain.py) | 円盤の基底、入力正規化層、スペクトル層、`LENO2d`、`DiskFNO2d` |

数式とモデル構造を確認する場合は `model_*.py` の基底作成、スペクトル層、
モデルの `forward` の順に読んでください。実験の流れは各実行入口の `main` から追えます。
モデル側は学習スクリプトをimportしないため、モデルだけを別の実験から利用できます。

```python
from model_squaredomain import LENO2d as SquareLENO2d
from model_circledomain import LENO2d as CircleLENO2d
```

各実行入口からも同じモデル名・基底関数をimportできます。
ファイル分割と改名に伴う演算式、既定の学習条件、`state_dict` のキーの変更はありません。
既存の学習スクリプトが出力したチェックポイントを継続して利用できます。
旧ファイル名を使っている実行コマンドやimport文は、新しい名前に更新してください。

## 領域ごとの機能

| | `squaredomain.py` | `circledomain.py` |
|---|---|---|
| 領域 | 矩形 | 円盤 |
| 入力状態 | `(B, Ny, Nx, Cin)` | `(B, N, 4)` と時間刻み `(B,)` |
| 出力状態 | `(B, Ny, Nx, 2)` | `(B, N, 2)` |
| Laplaceモデル | `LENO2d` | `LENO2d` |
| Laplaceスペクトル層 | `SpectralConvSeparable` | `SpectralConvNodes` |
| Fourierスペクトル層 | `SpectralConvFFT2d` | `SpectralConvFFT2d` |
| 円盤のFourierモデル | — | `DiskFNO2d` |
| 学習・評価 | `train_leno`, `eval_epoch` | 同左 |
| 履歴保存 | `save_training_history` | 同左 |
| 全時間テスト評価 | `evaluate_full_time_test` | 同左 |
| 1軌道の可視化 | `visualize_rollout_teacher_gm2d` | 同左 |

同名関数でも、領域に応じて基底・質量重み・データ引数は異なります。
両ファイルのモデルやチェックポイントを相互交換できるという意味ではありません。

## 実行

Python 3.10以降と、NumPy、SciPy、PyTorch、Matplotlibが必要です。
RAMの使用量表示には任意でpsutilを使用します。

```bash
python -m pip install numpy scipy torch matplotlib psutil
```

円盤データの例です。パスは実際の教師ファイルに置き換えてください。
`--teacher_paths` は複数パス・カンマ区切り・ワイルドカードに対応します。
FEM側もCPUメモリマップで読み込み、全ファイルのx/yを結合せず必要な遷移だけを取り出します。
`--teacher_mmap 0` で通常読み込みに切り替えられます。

```bash
python circledomain.py \
  --teacher_paths "data/disk/teacher_[0-9]*.pt" \
  --dt_scale 50 --train_fraction 0.8 --split_seed 0 \
  --save_dir outputs/disk --out_prefix leno_disk \
  --Nr 128 --Ntheta 256 --basis_type disk --K 128 \
  --width 64 --depth 4 --epochs 200 --batch_size 1 \
  --lr_schedule step50 --use_layer_scaling 0 --resume 0 \
  --teacher_rollout 1 --t_snap "T/2,T"
```

教師の `T` と `--dt_scale` を一致させてください。FEMの時間刻みは
`dt_physical`、`dt_normalized`、`dt`、旧入力チャネルの順に読み込みます。
単位はメタデータまたは保存時刻との差分から判定し、不明なら `--teacher_dt_units physical`
または `normalized` を指定します。円盤の座標と4チャネル入力＋独立dtは維持します。
FDMの矩形256×256・固定保存時刻に対するstrict検査は円盤には適用しません。

両方とも既定で全ファイルの軌道集合を80%/20%に分割します。
`--num_train_trajectories` / `--num_test_trajectories` で分割後の採用軌道数を指定できます
（`-1` は全候補）。`--split_seed` が同じなら選択も再現されます。
`--train_sample_ids` / `--test_sample_ids` は元のsample_id、または `source_id:sample_id`
をカンマ区切りで指定します。軌道を途中で切る `--max_train_samples` / `--max_test_samples`
の使い方はエラーになります。sample_id/time_indexがない旧データは保存時刻から
軌道順を復元できる場合に対応し、遷移単位のランダム分割は行いません。

矩形の直接教師データの例です。`--teacher_strict_standard 0` は固定実験仕様の検査を無効にします。
教師ファイルの時刻・境界条件・領域サイズは実際のデータに合わせて指定してください。

```bash
python squaredomain.py \
  --teacher_paths data/rectangle/teacher.pt \
  --teacher_strict_standard 0 \
  --save_dir outputs/rectangle --out_prefix leno_rectangle \
  --operator_type laplace --Kx 16 --Ky 16 \
  --width 64 --depth 4 --epochs 200 --batch_size 1 \
  --lr_schedule step50 --use_layer_scaling 0 --resume 0 \
  --teacher_rollout 1 --t_snap "50,250"
```

コードブロックの改行継続はbash用です。PowerShellでは1行にまとめて実行できます。
Colabでも4つのPythonファイルを同じ作業フォルダに置き、
`!python circledomain.py ...` または `!python squaredomain.py ...` と実行します。
`.py` ファイル自身には `%%writefile` を含めません。
引数の一覧は次のコマンドで確認できます。

```bash
python squaredomain.py --help
python circledomain.py --help
```

教師データはこのリポジトリには含まれていません。実験に使うデータを別途用意し、
上記の `--teacher_paths` を実際の保存先に置き換えてください。

## 教師データの基底再構成誤差

両スクリプトは既定で、学習前に設定した基底へ教師のu/vを射影し、
相対空間L2誤差 `||u-u_K||_M / ||u||_M` を計算します。
FEMは節点の質量重み、FDMは基底と同じ質量行列を使います。
評価対象は採用したtrain/testの全遷移です。入力状態 `x[..., :2]` と
次時刻の正解 `y[..., :2]` は別々に集計し、標準化や時間重みは適用しません。
各遷移を等重みに数えるため、同じ状態が隣接遷移の入力・正解に現れる場合もあります。

| オプション | 既定値 | 内容 |
|---|---|---|
| `--basis_reconstruction` | `1` | `0` で学習前の再構成評価を省略 |
| `--reconstruction_tol` | `0.01` | 許容相対L2誤差。0.01は1% |
| `--reconstruction_only` | `0` | `1` で再構成評価のみ実行し、モデル作成・学習前に終了 |

通常の実行引数に `--reconstruction_only 1` を追加すると、学習せず基底を調べられます。
`--basis_reconstruction 0` と併用した場合も評価を実行します。
FEMの `--inspect_only 1` は基底作成前に終了するため、評価時は指定しないでください。

保存先は `save_dir/out_prefix` に次の接尾辞を付けたCSVです。

- `_basis_reconstruction_curve.csv`: K=1から設定した実効基底数までの平均・最大相対L2誤差。
- `_basis_reconstruction_summary.csv`: 全設定基底での誤差、許容誤差、平均誤差が条件を満たす
  `min_K_mean`、全評価状態が条件を満たす `min_K_all`。設定範囲内で未達なら最小Kは空欄です。

CSVはsplit・input/target・解像度・u/v/combinedごとに出力します。
`combined` はu/vの二乗ノルムの和で計算します。誤差ゼロのゼロ場は相対誤差0とします。
全教師状態・両成分を1%以内に収めるには、各splitのinput/targetのu/v行で
`min_K_all` を確認し、その最大値を使います。空欄があれば基底数を増やして再評価してください。

Kは、**設定した基底に含まれるモードを固有値の昇順に並べた先頭K個**です。
同一固有値では元の列順を保ち、1列ずつ評価します。
FDMのKは `Kx_eff * Ky_eff` までの総モード数であり、
最小Kをそのまま `--Kx` や `--Ky` に指定する意味ではありません。
FEMの円盤基底は縮退モードを含むため、実効基底数が `--K` を上回る場合があります。
設定範囲外のモードは追加せず、モデルの基底や学習設定も自動変更しません。

対象はFEMのdisk/cartesian/読み込み節点基底と、FDMの
`--operator_type laplace`（Neumann・periodic）です。
Fourier FNO経路では通常学習時に理由を表示して評価を省略し、
`--reconstruction_only 1` では非対応をエラーで知らせます。
診断用基底だけをfloat64で質量直交化し、全基底による残差を直接計算してから
省略モードの二乗係数を加えるため、小さな誤差でも差し引きの桁落ちを抑えます。

基底再構成は、密な質量行列を使う独立の最小二乗解との比較で検証しています。

## 共通の変数名

| 名前 | 意味 |
|---|---|
| `xb`, `yb`, `yh` | 入力・教師出力・モデル予測のバッチ |
| `hist` | epochごとの学習履歴 |
| `out_prefix` | 保存先ディレクトリを含むファイル名の接頭辞。CLIの `args.out_prefix` は名前部分 |
| `opt`, `sched`, `scaler` | optimizer・学習率scheduler・AMP GradScaler |
| `amp_torch_dtype` | AMPで使用するPyTorchのdtype |
| `tr_u_sum`, `tr_v_sum`, `tr_n` | 学習誤差の累積値とサンプル数 |
| `loss_each`, `loss_u_each`, `loss_v_each` | 各サンプルのu/v相対L2 |
| `loss_weights` | 累積物理時間から計算した学習損失の重み |
| `t_cumulative_b` | 各遷移の目標時刻に対応する累積物理時間 |
| `dt_scale` | 入力の時間刻みを正規化・復元する尺度 |
| `dt_physical` / 層の `dt_scalar` | 物理単位の時間刻み |
| `spec`, `w1`, `fc0`, `fc1`, `fc2` | スペクトル層・点ごとの線形層・入出力の線形層 |
| `Phi`, `P`, `mu` | 円盤の固有関数・質量重み付き射影・Laplace固有値 |

時間依存層は両方とも既定で無効です。FEM側の時間依存層の数式は従来のままです。
格子と円盤では基底・質量重み・入力標準化が異なります。各領域のモデル定義と
実行入口に処理を分けており、同じ名前の関数でも引数やデータ形状は異なります。

## 検証

```bash
python squaredomain.py --smoke_test
```

このスモークテストは教師データを用意せずに実行できます。

今回の構成変更では、開発環境の既存34テストで小規模CPU学習、再開、
Fourierモデル、物理積分、出力形式、旧CLIと重みの互換性を確認しました。
さらに10通りのモデル設定で、分割前後の基底、予測値、勾配、Adam更新、
分割前のチェックポイントからの再開結果が一致することを確認しました。
検証環境はPythonとPyTorch 2.8.0+cpu、NumPy 2.2.6、SciPy 1.15.3です。
