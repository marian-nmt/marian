#!/usr/bin/env perl

use strict;
use POSIX;
use File::Temp qw/ tempfile tempdir /;

my $PID = $$;
$SIG{TERM} = $SIG{INT} = $SIG{QUIT} = sub { die; };

use Getopt::Long;
use File::Spec;

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
    "a|amunmt-bin-dir=s" => \$AMUNN_DIR,
    "m|moses-bin-dir=s" => \$MOSES_DIR,
    "s|scorer=s" => \$SCORER,
    "i|maximum-iterations=i" => \$MAX_IT,
    "d|dev=s" => \$DEV,
    "f=s" => \$SRC, 
    "e=s" => \$TRG,
    "o|decoder-opts=s" => \$DECODER_OPTS,
);

my $AMUNN = "$AMUNN_DIR/amun";
my $MIRA = "$MOSES_DIR/kbmira";
my $MERT = "$MOSES_DIR/mert";
my $EVAL = "$MOSES_DIR/evaluator";
my $EXTR = "$MOSES_DIR/extractor";

my $DEV_SRC = "$DEV.$SRC";
my $DEV_TRG = "$DEV.$TRG";

my $CONFIG = "--sctype $SCORER";

$WORK = File::Spec->rel2abs($WORK);

execute("mkdir -p $WORK");
execute("$AMUNN $DECODER_OPTS --show-weights > $WORK/run1.dense");
dense2init("$WORK/run1.dense", "$WORK/run1.initopt");

execute("rm -rf $WORK/progress.txt");
for my $i (1 .. $MAX_IT) {
    unless(-s "$WORK/run$i.out") {
        execute("cat $DEV_SRC | $AMUNN $DECODER_OPTS --load-weights $WORK/run$i.dense --n-best | perl -pe 's/\@\@ //g' > $WORK/run$i.out");
    }
    execute("$EVAL $CONFIG --reference $DEV_TRG -n $WORK/run$i.out | tee -a $WORK/progress.txt");
    
    my $j = $i + 1;
    unless(-s "$WORK/run$j.dense") {
        execute("$EXTR $CONFIG --reference $DEV_TRG -n $WORK/run$i.out -S $WORK/run$i.scores.dat -F $WORK/run$i.features.dat");
        
        my $SCORES = join(",", map { "$WORK/run$_.scores.dat" } (1 .. $i));
        my $FEATURES = join(",", map { "$WORK/run$_.features.dat" } (1 .. $i));
    
        execute("$MERT --sctype $SCORER --scfile $SCORES --ffile $FEATURES --ifile $WORK/run$i.initopt -d 9 -n 20 -m 20 --threads 20 2> $WORK/mert.run$i.log");

        log2dense("$WORK/mert.run$i.log", "$WORK/run$j.dense");
        dense2init("$WORK/run$j.dense", "$WORK/run$j.initopt");
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

sub log2dense {
    my $log = shift;
    my $dense = shift;

    open(OLD, "<", $log) or die "can't open $log: $!";
    open(NEW, ">", $dense) or die "can't open $dense: $!";
    
    my @weights;
    while(<OLD>) {
        chomp;
        if (/^Best point: (.*?)  =>/) {
            @weights = split(/\s/, $1);
        }
    }
    close(OLD) or die "can't close $log: $!";
    my $i = 0;
    foreach(@weights) {
        print NEW "F$i= ", $_, "\n";
        $i++;
    }
    close(NEW);
}

sub dense2init {
    my $dense = shift;
    my $init = shift;

    open(OLD, "<", $dense) or die "can't open $dense: $!";
    open(NEW, ">", $init) or die "can't open $init: $!";
    
    my @weights;
    while(<OLD>) {
        chomp;
        if (/^F\d+= (\S*)$/) {
            push(@weights, $1);
        }
    }
    close(OLD) or die "can't close $dense: $!";
    print NEW join(" ", @weights), "\n";
    print NEW "0 " x scalar @weights, "\n";
    print NEW "1 " x scalar @weights, "\n";
    close(NEW);
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
