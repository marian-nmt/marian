#!/usr/bin/env perl

use strict;
use POSIX;
use File::Temp qw/ tempfile tempdir /;

my $PID = $$;
$SIG{TERM} = $SIG{INT} = $SIG{QUIT} = sub { die; };

use Getopt::Long;

my $AMUNN_DIR = "";
my $MOSES_DIR = "";
my $DECODER_OPTS = "";

my $time = time();
my $WORK = "tuning.$time";
my $SCORER = "BLEU";

my $MAX_IT = 10;

my ($SRC, $TRG) = ("ru", "en");
my $DEV = "dev";

GetOptions(
    "w|working-dir=s" => \$WORK,
    "a|amunn-bin-dir=s" => \$AMUNN_DIR,
    "m|moses-bin-dir=s" => \$MOSES_DIR,
    "s|scorer=s" => \$SCORER,
    "i|maximum-iterations=i" => \$MAX_IT,
    "d|dev=s" => \$DEV,
    "f=s" => \$SRC, 
    "e=s" => \$TRG,
    "o|decoder-opts=s" => \$DECODER_OPTS,
);

my $AMUNN = "$AMUNN_DIR/bin";
my $MIRA = "$MOSES_DIR/kbmira";
my $EVAL = "$MOSES_DIR/evaluator";
my $EXTR = "$MOSES_DIR/extractor";

my $DEV_SRC = "$DEV.$SRC";
my $DEV_TRG = "$DEV.$TRG";

my $CONFIG = "--sctype $SCORER --filter /work/wmt16/tools/scripts/cleanBPE";

execute("mkdir -p $WORK");
execute("$AMUNN $DECODER_OPTS --show-weights > $WORK/run1.dense");
execute("rm -rf $WORK/progress.txt");
for my $i (1 .. $MAX_IT) {
    unless(-s "$WORK/run$i.out") {
        execute("cat $DEV_SRC | $AMUNN $DECODER_OPTS --load-weights $WORK/run$i.dense --n-best > $WORK/run$i.out");
    }
    execute("$EVAL $CONFIG --reference $DEV_TRG -n $WORK/run$i.out | tee -a $WORK/progress.txt");
    
    my $j = $i + 1;
    unless(-s "$WORK/run$j.dense") {
        execute("$EXTR $CONFIG --reference $DEV_TRG -n $WORK/run$i.out -S $WORK/run$i.scores.dat -F $WORK/run$i.features.dat");
        
        my $SCORES = join(" ", map { "$WORK/run$_.scores.dat" } (1 .. $i));
        my $FEATURES = join(" ", map { "$WORK/run$_.features.dat" } (1 .. $i));
    
        execute("$MIRA --sctype $SCORER -S $SCORES -F $FEATURES -d $WORK/run$i.dense -o $WORK/run$j.dense 2> $WORK/mira.run$i.log");
        normalizeWeights("$WORK/run$j.dense");
    }
    execute("cp $WORK/run$j.dense $WORK/weights.txt")
}

sub execute {
    my $command = shift;
    logMessage("Executing:\t$command");
    my $ret = system($command);
    if($ret != 0) {
        logMessage("Command '$command' finished with return status $ret");
        logMessage("Aborting and killing parent process");
        kill(2, $PID);
        die;
    }
}

sub normalizeWeights {
    my $path = shift;
    my ($temp_h, $temp) = tempfile();
    open(OLD, "<", $path) or die "can't open $path: $!";
    
    my @weights;
    my $sum = 0;
    while (<OLD>) {
        chomp;
        if (/^(F\d+) (.+)$/) {
            push(@weights, [$1, $2]);
            $sum += abs($2);
        }
    }
    close(OLD) or die "can't close $path: $!";
    foreach(@weights) {
        print $temp_h $_->[0], "= ", $_->[1]/$sum, "\n";
    }
    close($temp_h);
    rename($temp, $path) or die "can't rename $temp to $path: $!";
}

sub logMessage {
    my $message = shift;
    my $time = POSIX::strftime("%m/%d/%Y %H:%M:%S", localtime());
    my $log_message = $time."\t$message\n"; 
    print STDERR $log_message;
}

sub wc {
    my $path = shift;
    my $lineCount = `wc -l < '$path'` + 0;
    return $lineCount;
}
