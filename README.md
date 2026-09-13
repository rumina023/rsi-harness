# Bounded RSI Evaluation Harness / 実測する自己改善ハーネス

> **研究用の合成テストベッドです。AGI、継続的なrecursive
> self-improvement、または配備可能な自己改変を実証するものではありません。**
> 2026-09-14の12 seed・4世代アブレーションでは、recursive優位は事前登録済みの
> 統計基準を満たしませんでした。詳細は
> [RSI_TESTBED_RESULTS_2026-09-14.md](RSI_TESTBED_RESULTS_2026-09-14.md) を参照してください。

限定された合成探索課題で、提案・測定・採用/棄却・次世代への継承を行う研究プロトタイプです。汎用AGI、深い意味理解、継続的なRSIの達成を示すものではありません。

## 公開版の実行方法と制約

Python 3.10以上、Python標準ライブラリのみで動作します。まずこのディレクトリで `python -m unittest -v` を実行してください。`python anchor_verifier.py --benchmark` と `python blind_benchmark.py` もLLMへの接続なしで実行できます。

基本ループは認証済みCodex CLIを使い、`python harness.py --model YOUR_AVAILABLE_MODEL --generations 3` で実行します。利用可能なモデル名を指定してください。

生ログを含む `runs/` は公開対象から除外しています。段階探索とオフライン再開は過去レポートに依存しているため、新規取得した状態ではそのまま実行できません。`staged.py` と `resume_offline.py` 内の開始レポートのパスを、自分で生成した `harness.py` のレポートへ変更してください。過去実験と出発条件は異なります。

Anchor Verifierはルールベースの試作です。属性分解・再構成・反証候補の生成を行いますが、再構成できることは真理の証明ではありません。`anchor_gate.py` は数値比較から証拠を作る補助ゲートであり、独立した意味的検証を実証していません。基本ループの `harness.py` は数値ゲートを使い、Anchorゲートは `staged.py` に接続されています。

過去の実験概要は [結果.md](結果.md) にあります。テスト数などは当時の記録です。ライセンスは [MIT](LICENSE) です。

## 段階的な候補選別版

`python staged.py` で前回の改善済み設定から2ラウンド実行します。6提案を数値設定で重複排除し、8→32→80課題/カテゴリーで段階評価します。平均差に加えて不確実な候補の上側推定値で1枠を残します。最終候補と現行版から各1回の次世代提案を生成し、別課題で比較します。設定上は最大18回の提案呼び出しで、形式違反時には再試行があります。同時実行上限は1です。

現在の段階探索はローカルOllama (`http://localhost:11434`) の `zarigata/Qwen2.5-0.5B-Instruct:test` に固定されています。4モデルの一覧はありますが、4モデルを同時使用する実装ではありません。HTTPタイムアウトは1回500秒であり、実験全体の制限時間ではありません。`python ollama_smoke.py` で接続を試せます。小型モデルの形式違反によって停止する場合があります。

`python audit_staged.py runs/<staged-日時>` で保存されたシミュレーション値と選別・採用を再計算できます。前回の `runs/20260911-170813/report.json` が出発点として必要です。段階ごとに課題を分け、採用確認と最終評価も分離します。改善能力の測定は1段階だけであり、継続的なRSIを証明する実験ではありません。

Python標準ライブラリと認証済みCodex CLIで動作します。

```powershell
python -m unittest -v
python harness.py --model gpt-6-astra --generations 3
```

3役の独立したLLM呼び出しが探索ポリシーを提案し、開発課題で実測した結果を4番目の呼び出しが検討します。新しい検証課題で改善を確認した変更だけを採用し、次世代に探索設定・改善指示・過去の実験記録を渡します。各世代は4回、最大3世代で12回のモデル呼び出しです。各呼び出しは180秒でタイムアウトします。通常のアカウント利用枠を使います。

変更対象はJSONの探索設定と次世代向けの指示です。評価器や採用条件は固定し、モデル生成コードは実行しません。CLIは一時ディレクトリでread-only実行しますが、これは強固なOSレベルの情報隔離ではありません。プロンプトでツール使用を禁止し、出力イベントにツール使用があれば実験を中止します。

`runs/` にリクエスト、応答、イベント、前後の点数、採用状態、最終レポートを保存します。過去の仕事として参照するのは本実験の履歴だけです。基盤モデルの内部知識データベースへ直接アクセスする機能はありません。最終評価300課題は改善中には使用しません。

これは限定された合成探索課題での閉ループ実験です。ベースラインは意図的に単純なランダム探索です。通常のポリシー最適化を超えるRSIの証明には、改善指示を固定した対照群との同予算比較、複数回の独立実験、未知分野への汎化検証が追加で必要です。

## 改善能力を測るRSIテストベッド

`rsi_testbed.py` は、性能そのものではなく「過去の改善実験を使って次の改善を発見する能力」を反証可能に測る、オフラインの追加テストベッドです。

```powershell
python -m unittest -v
python rsi_testbed.py --seeds 12 --generations 4
python rsi_testbed.py --seeds 12 --generations 4 --ablations
```

各世代では Builder、Critic、Adversary、Judge、Research Memory、Meta-Optimizer を循環させます。可変なのは `policy`、`prompt`、`memory`、`tools`、`algorithm`、`agent_topology` に対応するデータのみです。セキュリティ境界、提案数と探索コール数の資源上限、開発Judge、最終external evaluator、採択規則は候補データから隔離され、実行中に境界ダイジェストで検査されます。

同一seed・同一計算予算で、改善記録を保持するrecursive群と、毎世代Research Memoryを空にするfrozen群を比較します。`P_t` と `I_t` を比較可能にするため、両群とも採択済みpolicy自体は現在の候補として保持します。frozen群には過去の成功・失敗・反証という改善知識を一切渡さず、Meta-Optimizerも固定です。世代ごとに `P_t`、検証済み改善量 `I_t=P_{t+1}-P_t`、ゼロ除算を除外した加速率 `A_t=I_{t+1}/I_t` を保存します。最終比較は未使用の固定external evaluatorに加え、31-bit・環状相互作用・deceptive blockという異なる地形の未知domain evaluatorでも行い、性能・robustness・falsification・generalization・cost efficiencyを別々に記録します。主解析はseed対応のブートストラップ区間と符号置換検定です。

`--ablations` は `memory-only`、`meta-only`、`algorithm-only` も追加します。metaを使う群には、最初の3世代だけ全員同一の事前登録済みschedule probeを与え、それ以後のscheduleだけを記憶された検証結果で選択します。`algorithm-only` は同じprobeを知識なしで固定巡回します。これにより、差が出たとしても「経験記憶」「経験で更新されたMeta-Optimizer」「知識を継承しない固定アルゴリズム変化」のどれが寄与したかを、frozen群との対応比較で切り分けます。

この試験は合成タスク、固定の候補生成規則、少数の探索カテゴリーに限られます。統計的な差が出ても、一般的なrecursive self-improvement、AGI、または配備可否の根拠にはなりません。

2026-09-14の12 seed・4世代アブレーション結果は [RSI_TESTBED_RESULTS_2026-09-14.md](RSI_TESTBED_RESULTS_2026-09-14.md) に記録しています。事前登録済みの判定基準ではrecursive優位は確認されませんでした。
