#!/bin/env perl

use strict;

use Getopt::Long;

my $AMUNN = "/home/marcinj/Badania/amunn/build/bin/amunn";
my $MOSES = "/data/smt/mosesMaster/bin";
my $DECODER_OPTS = "";

my $MIRA = "$MOSES/kbmira";
my $EVAL = "$MOSES/evaluator";
my $EXTR = "$MOSES/extractor";

my $time = time();
my $WORK = "tuning.$time";
my $SCORER = "BLEU";

my $MAX_IT = 10;

my ($SRC, $TRG) = ("ru", "en");
my $DEV = "dev";

GetOptions(
    "working-dir=s" => \$WORK,
    "scorer=s" => \$SCORER,
    "maximum-iterations=i" => \$MAX_IT,
    "dev" => \$DEV,
    "f=s" => \$SRC, 
    "e=s" => \$TRG,
    "decoder-opts=s" => \$DECODER_OPTS,
);

system("mkdir -p $WORK");

my $DEV_SRC = "$DEV.$SRC";
my $DEV_TRG = "$DEV.$TRG";

my $CONFIG = "--sctype $SCORER --filter /work/wmt16/tools/scripts/cleanBPE";

system("$AMUNN $DECODER_OPTS --show-weights > $WORK/run1.dense");

for my $i (1 .. $MAX_IT) {
    system("cat $DEV_SRC | $AMUNN $DECODER_OPTS --weights-file $WORK/run1.dense --n-best > $WORK/run$i.out");
    system("$EVAL $CONFIG --reference $DEV_TRG -n $WORK/run$i.out | tee -a $WORK/progress.txt");
    system("$EXTR $CONFIG --reference $DEV_TRG -n $WORK/run$i.out -S $WORK/run$i.scores.dat -F $WORK/run$i.features.dat");
    my $j = $i + 1;
    system("$MIRA --sctype $SCORER -S $WORK/run$i.scores.dat -F $WORK/run$i.features.dat -d $WORK/run$i.dense -o $WORK/run$j.dense");
    system("cp $WORK/run$i.dense $WORK/weights.txt")
}
