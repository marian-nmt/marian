`install.sh` is a helper script that downloads and compiles fastalign and extract-lex, and copies
required binaries into _./bin_.

Shortlist files (_lex.s2t_ and _lex.t2s_) can be created using `generate_shortlists.pl`, for
example:

    perl generate_shortlists.pl --bindir ./bin -s corpus.bpe.src -t corpus.bpe.tgt

