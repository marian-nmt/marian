# Using Marian with factors

Following this README should allow the user to train a model with source and/or target side factors. To train with factors, the data must be formatted in a certain way. A special vocabulary file format is also required, and its extension should be `.fsv` as providing a source and/or target vocabulary file with this extension is what triggers the usage of source and/or target factors. See details below.

### Requirements:

In order to use factors in Marian, you should use at least Marian 1.9.0 unless you want to use a factors functionality that requires setting one of the following command line options to their non default values: `--factors-combine`, `-—factors-dim-emb` and `--lemma-dependency` as they were only introduced after Marian 1.10.20+.

## Define factors

Factors should be organized in "groups," where each group represents a different feature. For example, there could be a group denoting capitalization and another denoting subword divisions.

Factors within a single group should start with the same string.

For example, for a capitalization factor group, the individual factors could be:

* `c0`: all lowercase

* `c1`: first character capitalized, rest lowercase

* `c2`: all uppercase

If there were a second factor group for subword divisions, the individual factors could be:

* `s0`: end of word, whitespace should follow

* `s1`: join token with next subword

There is no limit on the number of factor groups barring some practical limitations having to do with how the vocabulary is stored by `marian`. If the limit is exceeded `marian` will throw an error.

Factor group zero is always the actual words in the text, referred to as *lemmas*.

## Data preparation

Factors are appended to the *lemmas* with a pipe `|`. The pipe also separates factors of multiple groups.

Example sentence:

```
Trump tested positive for COVID-19.
```

Preprocessed sentence:
```
trump test@@ ed positive for c@@ o@@ v@@ i@@ d - 19 .
```

Apply factors:
```
trump|c1|s0 test|c0|s1 ed|c0|s0 positive|c0|s0 for|c0|s0 c|c2|s1 o|c2|s1 v|c2|s1 i|c2|s1 d|c2|s0 -|c0|s0 19|c0|s0 .|c0|s0
```


## Create the factored vocabulary

Factored vocabularies must have the extension `.fsv`. How to structure the vocabulary file is described below. If using factors only on the source or target side, the vocabulary of the other side can be a normal `json`, `yaml`, etc.

The `.fsv` vocabulary must have two sections:

1. **Factors**

    The factor groups are defined with an underscore prepended. The colon indicates which factor group each factor inherits from. `_has_c` is used in the definition of the words in the vocabulary (see #2 below) to indicate that that word has that factor group. The `_lemma` factor is used for the words/tokens themselves; this must be present.

    ```
    _lemma

    _c
    c0 : _c
    c1 : _c
    c2 : _c
    _has_c

    _s
    s0 : _s
    s1 : _s
    _has_s
    ```

2. **Lemmas**

    These are the vocabulary entries themselves. They have the format of `LEMMA : _lemma [_has_c] [_has_s]`. The `_has_X` should only apply to lemmas that can have an `X` factor anywhere in the data (which will likely be all of the tokens except `</s>` and `<unk>`).

    Examples:
    ```
    </s> : _lemma
    <unk> : _lemma
    , : _lemma _has_c _has_s
    . : _lemma _has_c _has_s
    the : _lemma _has_c _has_s
    for : _lemma _has_c _has_s
    ```


#### Other Requirements

Certain characters are used by the `.fsv` vocabulary that will have to be escaped/replaced in the data: `#:_\|`

The tokens in the factor vocabularies (`c0`, `c1`, `s0`, etc.) cannot be present in any of the *lemmas*.

### Full `.fsv` file

Putting everything together, the final `.fsv` file should look like this. It can have comments (lines started by `#`).

 ```
 # factors

_lemma

_c
c0 : _c
c1 : _c
c2 : _c
_has_c

_s
s0 : _s
s1 : _s
_has_s

 # lemmas

</s> : _lemma
<unk> : _lemma
, : _lemma _has_c _has_s
. : _lemma _has_c _has_s
the : _lemma _has_c _has_s
for : _lemma _has_c _has_s
 ```

## Training options

There are two choices for how factor embeddings are combined with *lemma* embeddings: summation and concatenation.

```
--factors-combine TEXT=sum      How to combine the factors and lemma embeddings.
                                Options available: sum, concat
```

The dimension of the factor embeddings must be specified if using combine option `concat`. If using `sum`, the factor embedding dimension matches that of the lemmas.

```
--factors-dim-emb INT           Embedding dimension of the factors. Only used if concat is selected as factors combining form
```

Note: At the moment `concat` is only implemented for usage in the source side.

### Prediction

If using factors on the target side, there are multiple options for how factor predictions are generated related to the form of conditioning / dependencies of factors and lemmas. If no option is set with `--lemma-dependency`, the default behavior will be predicting the factors with no lemma dependency.

```
--lemma-dependency TEXT         Lemma dependency method to use when predicting target factors.
                                Options: soft-transformer-layer, hard-transformer-layer, lemma-dependent-bias, re-embedding

--lemma-dim-emb INT=0           Re-embedding dimension of lemma in factors
```

* `soft-transformer-layer`: Uses an additional transformer layer to predict the factors using the previously predicted lemma
* `hard-transformer-layer`: Like `soft-transformer-layer` but with hard-max
* `lemma-dependent-bias`: Adds a learned bias term based on the predicted lemma to the logits of the factors. There is no additional transformer layer introduced with this option
* `re-embedding`: After predicting a lemma, re-embed the lemma and add this new vector before predicting the factors
* `lemma-dim-emb`: Controls the dimension of the re-embedded lemma when using the option `re-embedding`


### Weight tying

If you use factors both on the source and target side, and the factors are the same for both sides you can tie the embeddings exactly as you do for non factored models.

If factors are used only on one side (either source or target) with a joint vocabulary, there are two options for tying source and target embedding weights:

1. Use combine option `concat` (If using factors only on the source side).
2. Use combine option `sum`, and create "dummy" factors on the non-factorized side. This entails creating a factored vocabulary where the same number of factors are present as are on the side with meaningful factors. In the previous example, if we have the capitalization and subword factors on the source side, the target side would have five different dummy factors (they can all be in the same group). In the *lemma* section of the `.fsv` file we would just not put `_has_X` for any lemma.

    ```
    # factors

    _lemma

    _d
    d0 : _d
    d1 : _d
    d2 : _d
    d3 : _d
    d4 : _d
    _has_d

    # lemmas

    </s> : _lemma
    <unk> : _lemma
    , : _lemma
    . : _lemma
    le : _lemma
    pour : _lemma
    ```

## Examples
Some examples of possible commands to train factored models in marian:
* Using factors on both source and target. Using `sum` to combine lemma and factor embeddings. No tied embeddings and no lemma dependency when predicting the factors:
```
path_to/build/marian -t corpus.fact.{src,trg} \
                     -v vocab.{src,trg}.fsv
```
* Using factors only on the source side. Using `concat` to combine lemma and factor embeddings. Source, target and output embeddings matrices tied:
```
path_to/build/marian -t corpus.fact.src corpus.trg \
                     -v vocab.src.fsv vocab.trg.yml \
                     --factors-combine concat \
                     --factors-dim-emb 8 \
                     --tied-embeddings-all
```
* Using factors only on the target side. Using `sum` to combine lemma and factor embeddings. Target and output embedding matrices tied. Predicting factors with `soft-transformer-layer` lemma dependency:
```
path_to/build/marian -t corpus.src corpus.fact.trg \
                     -v vocab.src.yml vocab.fsv.trg \
                     --tied-embeddings \
                     --lemma-dependency soft-transformer-layer
```
