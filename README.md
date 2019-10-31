Game Programming Workshop 2019 投稿論文 「グリッド世界を用いた階層型強化学習の評価」 実験コード  
under construction

# Usage
1. 必要なパッケージのインストール  
以下が著者の環境  
  - python 3.7.3
  - chainer 6.0.0
  - chainerrl 0.6.0
  - numpy 1.16.2
  - gym 0.12.5
  - optuna 0.15.0

2. 実行
train_a3c_gym.pyを動かすことで実行できる.


実行時に引数を追加することで色々なオプションをつけて実行できる

例(ゴールを1マス手前で隠す環境をlstmでやる)

`python train_a3c_gym.py 16 --logger-level=30 --hidden_dist=1 --lstm`

3. 結果の確認
プログラムを動かすとresultファイルが作られるので, その下にいって確認する.
