# llmcpp

## 概要
ローカル環境で生成AIによりテキストや画像、音声を生成する作業を支援する CLI のフロントエンドである。

* 想定するバックエンド
	* テキスト生成
		* oobabooga/text-generation-webui (https://github.com/oobabooga/text-generation-webui)
		* lostruins/koboldcpp (https://github.com/lostruins/koboldcpp)
	* 画像生成
		* AUTOMATIC1111/stable-diffusion-webui (https://github.com/AUTOMATIC1111/stable-diffusion-webui)
	* 音声生成
		* litagin02/Style-Bert-VITS2 (https://github.com/litagin02/Style-Bert-VITS2)

それぞれのバックエンドが提供する GUI を手動で操作する手間から解放し、生成する対象に依存しない統一的なインターフェースを提供し、創作活動を支援する。

## 機能
* 再帰下降構文解析により実装された堅牢なマクロ機能による、プログラミング可能なプロンプト生成を提供する。
* プロンプトや生成されたテキストを任意の複数のファイルに分割して管理する。(コードブロックによるファイル分割、 `file` マクロ)
* 小説の執筆のように、過去に生成されたテキストが蓄積し肥大化する作業において、有限で貧弱なコンテキストに収まるようプロンプトを構成する。(`head`, `tail`, `head_tail` マクロ)
* テキストを生成する工程を自然言語による記述で構造化する。(`phase` マクロ)
* LLM のトークナイザーが通信により返したトークン数の情報をローカルにキャッシュし高速に再利用する。
* スタックトレースを含むデバッグに有益な情報を、重要度に応じてフィルタリング可能な形態でログ出力する。
* 主に Windows 環境のように UTF-8 をターミナルの標準のコードページとして採用していない OS 環境に対して、文字コードを UTF-8 に統一して扱う。
* VRAM 容量の制限に起因して同時に複数のサーバーを起動することができないローカル環境にて、統一的な方法で各サーバーを起動・終了する。

## ビルド要求
* Boost 1.88.0+
	* https://www.boost.org/
* C++20+

```
vcpkg install boost-beast:x64-windows-static boost-asio:x64-windows-static boost-program-options:x64-windows-static boost-multi-index:x64-windows-static boost-log:x64-windows-static boost-nowide:x64-windows-static boost-stacktrace:x64-windows-static boost-exception:x64-windows-static boost-algorithm:x64-windows-static boost-date-time:x64-windows-static boost-serialization:x64-windows-static boost-url:x64-windows-static boost-process:x64-windows-static boost-spirit:x64-windows-static boost-fusion:x64-windows-static
vcpkg integrate install
```

## 共通設定
下記の設定は、対象とするバックエンドに依存せず共通して参照される。

| オプション | 概要 | デフォルト値 | 備考 |
| --- | --- | --- | --- |
| --mode | モード | tg | `(tg\|kc\|sd\|sb)` のいずれか |
| --base-path | 各種パスの基準となるパス | . |  |
| --log-level | ログレベル | info | `(trace\|debug\|info\|warning\|error\|fatal)` のいずれか |
| --log-file | ログファイル | log |  |
| --config-file, -c | コマンドラインオプションを指定した設定ファイル | config.ini |  |
| --verbose, -v | ログ出力をコンソールに冗長に出力する | true |  |
| --number-iterations, -N | 処理を反復する回数 | 1 | `-1` を指定すると実行を停止するまで無限に処理を反復する。 |
| --define, -D | マクロ | "" | 詳細は後述 |
| --phases | phase | "" | 詳細は後述 |

`config-file` で指定した設定ファイルは、INI 形式を想定した構文解析により読み込まれる。コマンドラインと設定ファイルの双方が共通するコマンドラインオプションを指定している場合、コマンドラインで指定された方を優先する。頻繁に指定するコマンドラインオプションは設定ファイルで指定することが検討できる。

```ini:config.ini
# ポート番号の設定
llm-port = 5001

# 真偽値オプションの設定例
verbose = true

# Gemma 4 向けの設定例
llm-generation-prefix = <|channel>thought\n
kc-stop-sequence = <turn|>
kc-temperature = 1.0
kc-top-k = 64
kc-top-p = 0.95
kc-max-context-length = 8192
kc-max-length = 4096
llm-reasoning-prefix = <|channel>thought\n
llm-reasoning-suffix = <channel|>
```

モード名は下記の通り、それぞれと対応するバックエンドの略称である。

* tg: Text-Generation-webui
* kc: KoboldCpp
* sd: Stable-Diffusion-webui
* sb: Style-Bert-VITS2

パスの区切り文字は `/` でなければならない。ファイル名の拡張子は省略することができる。テキストファイルのパスが期待される文脈でパスに拡張子が含まれない場合、 `.txt` を補完する。相対パスは `--base-path` オプションで指定したパスを基準として解釈される。

## oobabooga/text-generation-webui の使用法
oobabooga/text-generation-webui(https://github.com/oobabooga/text-generation-webui) を導入し、下記の準備をする。
1. `text-generation-webui/user_data/CMD_FLAGS.txt` にて ` --api` オプションを指定する。
2. `start_windows.bat` など実行環境と対応するスクリプトを実行し、サーバーを起動する。

`--mode tg` と `--llm-` および `--tg-` から始まるオプションを適宜指定して llmcpp を実行する。
使用するモデルはバックエンドが対応している形式であれば何でも構わないが、例として [Gemma 4](https://ai.google.dev/gemma/docs/core?hl=ja) などが挙げられる。

主要ファイル

| オプション | 概要 | デフォルト値 |
| --- | --- | --- |
| --llm-prompt-file | プロンプトファイル | prompt.txt |
| --llm-output-file | 出力ファイル | output.txt |

必要に応じて下記のオプションで通信先を指定する。oobabooga をデフォルトの設定で運用している場合、明示的に指定する必要はない。
プロンプトとは、例えば以下のようなテキストである。

```
<|turn>system
あなたは小説家です。内容の続きを500文字程度だけ執筆してください。

# 登場人物
## (中略)

# 直近の内容
{{include_tail:output,1024}}
<turn|>
<|turn>user
<turn|>
<|turn>model
```

既存のファイルの内容ではなく特定の文字列をプロンプトして使用する場合、 `--*-prompt-file` オプションではなく `--*-prompt` オプションを使用する。
`--*-prompt` オプションが指定されている場合、`--*-prompt-file` で指定されたファイルの内容は無視される。

| オプション | 概要 | デフォルト値 |
| --- | --- | --- |
| --llm-host | ホスト | localhost |
| --llm-port | ポート | 5000 |

実行を開始すると、 llmcpp は下記のように LLM に渡すプロンプトを作成する。

1. `prompt.txt` の内容をプロンプトに追加する。
2. `--llm-generation-prefix` オプションで指定された文字列をプロンプトに追加する。(デフォルトは空文字)

プロンプトを作成した後、LLM と通信し、テキストを生成する。生成されたテキストの先頭には `--llm-generation-prefix` オプションで指定された文字列が含まれる。
デフォルトでは、生成したテキストは `output.txt` の末尾に追加される。
変更する場合は `--llm-output-file` で出力先を指定する。親フォルダが存在しない場合、自動的に作成される。

LLMに小説を生成させる場合、`output.txt` は作成された小説になる。
LLMとチャットをする場合、`output.txt` は会話の履歴になる。

## lostruins/koboldcpp の使用法
lostruins/koboldcpp(https://github.com/lostruins/koboldcpp) を導入し、下記の準備をする。
1. ` --port 5000` オプションなどを指定して `koboldcpp.exe` を起動する。

`--mode kc` と `--llm-` および `--kc-` から始まるオプションを適宜指定して llmcpp を実行する。

```bat:run.bat
rem 起動用バッチの例
@echo off
cd /d %~dp0
set model=models/gemma-4-E4B-it-Q8_0.gguf

koboldcpp.exe --model %model% ^
  --gpulayers 99 ^
  --contextsize 8192 ^
  --usecuda ^
  --jinja ^
  --port 5000
```

## AUTOMATIC1111/stable-diffusion-webui の使用法
AUTOMATIC1111/stable-diffusion-webui (https://github.com/AUTOMATIC1111/stable-diffusion-webui) を導入し、下記の準備をする。

1. `webui-user.bat` など実行環境と対応するスクリプトを編集し、`COMMANDLINE_ARGS` に `--api` を追加する。他のサーバーに割り当てるポートを考慮して適宜 `--port 7861` 等を追加する。
2. `webui-user.bat` など実行環境と対応するスクリプトを実行し、サーバーを起動する。

`--mode sd` と `--sd-` から始まるオプションを適宜指定する。
下記のようなコマンドで画像を生成する。

`llmcpp --mode sd --sd-port 7861 --sd-width 832 --sd-height 1216 --sd-step 30 --sd-prompt "sky"`

## litagin02/Style-Bert-VITS2 の使用法
litagin02/Style-Bert-VITS2 (https://github.com/litagin02/Style-Bert-VITS2) を導入し、下記の準備をする。

1. `config.yml` を編集し、他のサーバーに割り当てるポートを考慮して適宜 `port: 5001` のように変更する。
2. `Server.bat` を実行し、サーバーを起動する。

`--mode sb` と `--sb-` から始まるオプションを適宜指定する。
下記のようなコマンドで音声を生成する。

`llmcpp --mode sb --sb-port 5001 --sb-model-name "amitaro" --sb-speaker-id 0 --sb-language JP --sb-text "こんにちは"`

### 共通オプション
## 反復
`-N` オプションにより処理を反復する回数を指定することができる。デフォルトの値は `1` である。`-1` を指定すると実行を停止するまで無限に処理を反復する。

## マクロ
囲まれたマクロを以下の文脈で使用することができる。

* 各種テキストファイルの内容
* コマンドラインオプションで指定するファイルパス

マクロは特定の文字列を `{{` と `}}` で囲むことにより記述する。
マクロには、 `{{foo(arg1, arg2, arg3, ...)}}` 形式のマクロ関数と、 `{{foo}}` 形式のマクロ変数がある。
マクロ関数の引数として文字列定数を指定する場合 `"arg"` というように `"` で囲む。
マクロは実行時の反復毎、pahse の実行毎に展開される。
ただし、`--log-file` で指定されたログファイルのパスは、プログラムの開始時に一度だけマクロを展開され、以降その時点のパスを使用し続ける。
これはプログラムがログファイルのストリームを開いたまま保持し効率的にログ出力するための制限である。

マクロ変数は `--define` オプションにより事前に定義することができる。
`--define "user={{user}}" "char={{char}}"` というようにマクロの展開先にマクロを指定することができる。
ただし上記のようにマクロの展開後の文字列が展開前の文字列と完全に一致した場合、再帰的なマクロの展開は中断される。
このような冗長なマクロの定義は、マクロを含むプロンプトを LLM を使用して生成する際に、マクロの展開を抑止するために役立つ。

### `file(path)`
指定されたテキストファイルの内容に展開される。
これによりプロンプトを複数のファイルに分割して管理することができる。

```
{{file("output")}}
{{file("output.txt")}}
{{file("parent_dir/output.txt")}}
```

### `head(str, max_token)`
指定された文字列のうち、先頭から指定されたトークン数分の文字列に展開される。
例えば最古の出力をプロンプトに含めることができる。

```
{{head(file("output"), 1024)}}
```

### `tail(str, max_token)`
指定された文字列のうち、末尾から指定されたトークン数分の文字列に展開される。
例えば最新の出力をプロンプトに含めることができる。

```
{{tail(file("output"), 1024)}}
```

### `head_tail(str, head_max_token, tail_max_token)`
指定されたテキストファイルの内容のうち、先頭と末尾それぞれから指定されたトークン数分の文字列に展開される。
先頭と末尾の間は `...` で表現される。
例えば最古と最新の出力をプロンプトに含めることができる。
```
{{head_tail(file("output"), 512, 512)}}
```

### `json_literal(str)`
生改行を `\n` に置換するなど、JSON の文字列リテラルとして正しくなるようエスケープする。

```
{ "key": "{{json_literal(file("filename"))}}" }
```

### `env(var)`
環境変数 `var` の値に展開される。
```
{{env("path")}}
```

### `datetime`
`yyyyMMddhhmmss` 形式で表現された実行時点の時刻に展開される。主に生成したテキストを生成単位で出力する目的で使用する。

### `N`
1から開始する現在の反復回数に展開される。主に生成したテキストを反復単位で出力する目的で使用する。

### `stdin`
標準入力から渡された文字列に展開される。

### `generated(subprompt)`
サブプロンプトを実行した結果に展開される。

### `let(name, value)`
マクロ変数を定義する。定義されたマクロは同ファイル、および同ファイルから `generated` マクロで呼び出されたサブプロンプト内でのみ有効である。

### `exec(exe, args ...)`
子プロセスとして実行ファイルを実行した結果に展開される。終了コードはマクロ変数 `{{exit_code}}` に設定される。

```
{{exec("cmd", "/c", "dir")}}
exit_code={{exit_code}}
```

### `summary(prompt, target, max_token)`
文字列を最大トークン数以下に要約する。

```
{{summary(file("summary"), file("long_file"), 1024)}}
```

`summary.txt` とは下記のようなファイルである。

```
<|turn>system
<|think|>
Summarize the following text in {{max_token}} tokens or fewer. Output only the summary with no introductory text, markdown formatting, or additional commentary.
<turn|>
<|turn>user
{{target}}
<turn|>
<|turn>model
```

`{{target}}`, `{{max_token}}` は `prompt` の実行中のみ設定される。展開語の文字列のトークン数は `max_token` に切り詰められる。


### `random(min, max)`
ランダムな整数値の10進数表記に展開される。引数は後ろ側から省略可能。`min` のデフォルト値は `0`、`max` のデフォルト値は `2147483647` である。`{{random()}}` で 32bit 整数値の表現範囲のうち正の値のみに展開される。例えば seed 値を実行単位で変更することができる。

### `choice(args ...)`
任意の数の文字列からランダムにひとつ選択して展開する。

```
{{choice("foo", "bar")}}
```

### `phase`
1回の処理は1個以上の phase から構成され、それらは順番に実行される。
phase は `--phases "MyPhase1" "MyPhase2" "MyPhase3"` オプションで任意の数だけ指定可能である。 
現在実行中の phase は、プロンプトの内容に `{{phase}}` マクロを記述することにより、実行時に参照できる。
前の phase は `{{prev_phase}}` マクロ、次の phase は `{{next_phase}}` マクロで参照できる。

例えば `--phases "{{char}}" "{{user}}" --llm-generation-prefix "{{phase}}: "` オプションを指定した場合、pahse として現在発言している人物の名前を扱い、それをプロンプトの先頭に追加することにより、LLM が当該の人物の発言を生成するように誘導する。
`prompt.txt` の内容に phase の要素に関する説明を含めることにより、1回の処理を複数の工程に分割し、動的に生成の方法を制御することができる。

`--llm-paragraph-file` オプションで段落ファイルを指定した場合、段落ファイルに含まれる内容が phase に割り当てられる。
段落ファイルは、順序リストあるいは非順序リストにより段落の名前を記述し、その直後にインデントされた順序リストあるいは非順序リストによりその段落の説明を記述した、マークダウン記法に準拠したファイルである。下記は段落ファイルの具体例である。

```
1. 段落1の名前
    * 説明1
    * 説明2
    * 説明3
2. 段落2の名前
    * 説明1
    * 説明2
    * 説明3
3. 段落3の名前
    * 説明1
    * 説明2
    * 説明3
```

具体的な利用例として、`prompt.txt` に下記のように記述すれば、現在の段落と前後の段落に関する情報をプロンプトに含めることができる。

```
# 段落の情報
## 現在の段落
{{phase}}

## 前の段落
{{prev_phase}}

## 次の段落
{{next_phase}}
```

これにより、LLM が物語として一貫し、秩序正しいテキストを生成するよう誘導することができる。

## コードブロック単位のファイル出力
`--llm-code-block-extract` オプションを指定することにより、LLM の出力に含まれる markdown 形式のコードブロックをそれぞれファイルとして出力することができる。ファイル名は拡張子を省略して記述し、実行時に `.txt` を補完して解釈される。

````
```foo
aaa
bbb
ccc
```

```bar
aaa
bbb
ccc
```

```stdout
aaa
bbb
ccc
```
````

ただし、 `stdout` が指定された場合、そのコードブロックの内容はファイルではなく標準出力に出力される。
これらの機能は、LLM の出力を複数に分割し、それぞれを再利用するために利用することができる。
コードブロックを出力するためのプロンプトとは、例えば以下のようなものである。

````
# 命令
後述する出力形式に従い回答をコードブロックとして出力せよ。コードブロックの名前は厳密にそのまま利用せよ。

# 出力形式
```output\{{datetime}}
(中略)
```
````

## サーバーの起動と終了

テキストや画像、音声などを横断的に生成したい場合、以下のコマンドラインオプションで同期的にサーバーを起動あるいは終了することができる。現在は koboldcpp を対象として想定している。その他のサーバーにも対応する予定。

### サーバーの起動
```
llmcpp --create-process --server-executable-file koboldcpp.exe --server-arguments "--model %model% --port 5001" --server-host localhost --server-port 5001 --server-max-retries 60 --server-wait-ms 1000
```

`--create-process` が指定された場合、サーバーを起動するコマンドを非同期に実行後、`--server-host` と `--server-port` により指定された IPアドレス およびポートと通信が可能になるまで待機する。
`--server-host localhost`, `--server-max-retries 60`, `--server-wait-ms 1000` は省略することができる。

### サーバーの終了
```
llmcpp --terminate-process --server-executable-file koboldcpp.exe
```
`--terminate-process` が指定された場合、指定された実行ファイルのフルパスあるいはファイル名を利用してサーバーのプロセスを終了する。