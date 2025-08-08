# llmcpp

## 概要
llmcpp は、oobabooga/text-generation-webui(https://github.com/oobabooga/text-generation-webui) により提供される Open-API 互換の Web API を利用して、ローカル環境でテキスト生成する作業を支援する CLI のフロントエンドである。

## 事前準備
1. `text-generation-webui/user_data/CMD_FLAGS.txt` にて ` --api` オプションを指定する。
2. `start_windows.bat` など実行環境と対応するスクリプトを実行し、サーバーを起動する。

## チュートリアル
llmcpp はデフォルトで下記のファイルを読み込む。

* system_prompts.txt
* examples.txt
* history.txt

ログはデフォルトで下記に出力される。

* log.txt

ログをコンソールに冗長に出力する場合、`-v` オプションを指定する。

```
> llmcpp -v
```

ログに出力する情報を範囲を変更する場合、 `--log-level` オプションに続き、`(trace|debug|info|warning|error|fatal)` のいずれか指定する。デフォルトでは `info` が使用される。

必要に応じて下記のオプションで通信先を指定する。oobabooga をデフォルトの設定で運用している場合、明示的に指定する必要はない。

* `--host`
* `--port`
* `--api-key`

実行を開始すると、 llmcpp は下記のように LLM に渡すプロンプトを作成する。

1. system_prompts.txt の内容をプロンプトに追加する。
2. history.txt の内容を、末尾の行から優先して可能な限り、本来の順序でプロンプトに追加する。
3. examples.txt の内容を、可能な限り、本来の順序でプロンプトに追加する。

LLM に渡すプロンプトは、
ここでいう可能な限りとは、 `--max-total-context_tokens` オプションで指定しているコンテキストの最大トークン数(デフォルトの値は 4096)から、`--max-tokens` オプションで指定している LLM により生成されるテキストの最大トークン数(デフォルトの値は 512)を差し引いたトークン数を超過しない限りを意味する。
つまり llmcpp は、「プロンプト」と「LLM により生成されるテキスト」を足した全体のトークン数が、コンテキストの最大トークン数以下に収まるよう、`history.txt` や `examples.txt` から読み込むテキストの量を調整する。