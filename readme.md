# はじめに

Python の機械学習向け数値計算ライブラリ TensorFlow を Swift でも使用することができる [Swift for TensorFlow](https://www.tensorflow.org/swift) を試してみました．

## Swift for TensorFlow とは

既存の TensorFlow のラッパーを想像するかもですが，Swift for TensorFlow はそうではなく，Swiftコンパイラを拡張してTensorFlowを使えるようにしたものです．

- Swiftのラッパーライブラリではない
- Swiftコンパイラを拡張したもの

また，Swiftの機能でPythonを使用することができるので，Pythonの豊富な数値計算ライブラリを使用できます．


# インストール

公式サイトの [Install Swift for TensorFlow](https://github.com/tensorflow/swift/blob/master/Installation.md) に従えばインストールできます．なお，最近Swift 5がリリースされましたが，最新版の Xcode では上手くできず，Xcode 10.1 (Swift 4.2) で確認しました．


## Google Colab を使う

ローカル環境でインストール可能ですが，Swiftコンパイラ拡張のため，普段macOSでiOS開発を行っていると，標準コンパイラと拡張コンパイラの管理が面倒になります．手軽に試したいのであれば，Google Colab を使用した方がおすすめです（ただし，補完などの補助機能がXcodeより貧弱なのが辛い）．

これも公式の [Using Swift for TensorFlow](https://github.com/tensorflow/swift/blob/master/Usage.md) にて，Swiftが実行可能な [blank Swift notebook](https://colab.research.google.com/github/tensorflow/swift/blob/master/notebooks/blank_swift.ipynb) が公開されているので，それを自身のドライブにコピーすれば，すぐに使えます．

## ローカル環境での注意点

ちょっと触った感じから，気を付けておいた方が良いことです．

### 開発対象

PlaygroundおよびmacOS向けを選択します．

### Python

Swiftで呼び出せるPythonはOS標準のPythonが選択される．

```Swift
import Python
print("Python version: \(Python.version)") // 2系
```

通常は2系となる（pyenv等で3系を入れていても）．3系を使う場合は，[Python公式サイト](https://www.python.org/downloads/)から3系をダウンロードしてインストールしなければならない．

# SwiftでPythonを使う

Swift for TensorFlow を使う前に，Swift 自体の機能で Python を使ってみました．Python の豊富な数値計算ライブラリを使用できます．SwiftでNumPyが動くのは，何か不思議な感じです．

```Swift
import Python

// 私用環境に2系や3系が混在してる場合は指定しておく
PythonLibrary.useVersion(3, 6)

let np = Python.import("numpy")

let a = np.array([[1, 0], [0, 1]])
let b = np.array([[2], [3]])
let c = np.matmul(a, b)

print(c)
```

# Swift For TensorFlow を使ってみた

Swift For TensorFlow で簡単な機械学習をやりました．XOR を簡単な NN で学習します．Keras もしくは TensorFlow2 を触っていると，あまり違和感なく書けると思います．

ちなみにSwiftの仕様から ```let 𝛁model = hogehoge``` のように変数に数学記号を入れることができます．数学的にわかり易いですが，ちょっと開発環境のエディタとかで問題ないか心配なところもあります．

```Swift
import Foundation // Dateを呼ぶため
import TensorFlow

// XORのモデル
struct XOR: Layer {
  
  var layer1: Dense<Float>
  var layer2: Dense<Float>
  
  init(hiddenSize:Int = 2){
    self.layer1 = Dense(inputSize: 2,
                        outputSize: hiddenSize,
                        activation: sigmoid)
    self.layer2 = Dense(inputSize: hiddenSize,
                        outputSize: 1, 
                        activation: sigmoid)
  }
  
  @differentiable
  func applied(to input: Tensor<Float>, in context: Context) -> Tensor<Float> {
    return input.sequenced(in: context, through: layer1, layer2)
    }
}

let x: Tensor<Float> = [[0, 0], [0, 1], [1, 0], [1, 1]]
let y: Tensor<Float> = [[0], [1], [1], [0]]

var model = XOR()
let optimizer = SGD<XOR, Float>(learningRate: 0.1)
let context = Context(learningPhase: .training)

var date0 = Date()
for _ in 1...5000 {
  let dmodel = model.gradient { m -> Tensor<Float> in
    let t = m.applied(to: x, in: context)
    let loss = sigmoidCrossEntropy(logits: t, labels: y)
    return loss
  }
  
  optimizer.update(&model.allDifferentiableVariables, along: dmodel)
  
}

let date1 = Date().timeIntervalSince(date0)
print("elapsed_time: \(date1) sec")

let inference = round(model.inferring(from: x))
// print(inference)

for i: Int32 in 0 ..< 4 {
  print("x: \(x[i]), y: \(y[i]), inference: \(inference[i]), result:\(y[i] == inference[i] )")
}

print(model)
```

## パフォーマンスは？

上記の XOR.swift を Google Colab (CPU) で計算したところ，13秒程度で終わりました．比較として，Python + TensorFlow2 (Keras) でも同様なモデルと学習方法でXORを解いてみました．


```Python
import time
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import SGD

def xor():

  model = Sequential([
        Dense(input_dim=2, units=2),
        Activation("sigmoid"),
        Dense(input_dim=2, units=1),
        Activation("sigmoid")
    ])
  
  model.compile(loss="binary_crossentropy", optimizer=SGD(lr=0.1))
  
  X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
  Y = np.array([[0], [1], [1], [0]])

  batch_size = 4

  start = time.time()

  # 学習
  model.fit(X, Y, epochs=5000, batch_size=batch_size, verbose=0)

  elapsed_time = time.time() - start
  print ("elapsed_time: {0} sec".format(elapsed_time))
  
  # 確認
  classfied = model.predict_classes(X, batch_size=batch_size)
  prob = model.predict_proba(X, batch_size=batch_size)

  for (x, y, p, c) in zip(X, Y, prob, classfied):
    print("x = {}, prob = {}, classfied = {}, result = {}".format(x, p, c, y == c))
    
xor()
```

Python版は約6秒でした．コンパイラ言語であるSwiftの方が早いかと思いましたが，今回はPython版の方が早かったです．理由は分かりませんが，複雑な反復処理を行っていない，Swiftは複数のライブラリの読み込みがネックになった？のかなと．


# まとめ

Swift For TensorFlowを使用して，SwiftでもTensorFlow APIを使用して機械学習を実装できました．


## メリット

iOSアプリ開発者なら慣れたSwiftを使うことができる．

## デメリット

TensorFlow自体の更新頻度が高いうえに，Swiftもアップデートの頻度が高いので，安定性があるかは怪しいところです．また，入門エントリーが公式ぐらいしかないので，ちょっと調べ物は大変です．



# 参考

- [Swift for TensorFlowを触る - 三日坊主のプログラミング日誌](https://pgm-diary.hateblo.jp/entry/hello_s4tf)
- [今、僕が一番注目している Swift の新機能について、 iOSDC Japan 2018 で話します - koherent.org](https://koherent.org/iosdc/2018-trailer)
- [jupyter notebook - Swift kernel in Google Colaboratory - Stack Overflow](https://stackoverflow.com/questions/54015543/swift-kernel-in-google-colaboratory)
- [tensorflow/swift-models: Models and examples built with Swift for TensorFlow](https://github.com/tensorflow/swift-models)