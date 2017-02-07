#!/usr/bin/env perl

use strict;
use Getopt::Long;
use FindBin qw($Bin);
use File::Temp qw(tempdir tempfile);
use POSIX;

my $PID = $$;
$SIG{TERM} = $SIG{INT} = $SIG{QUIT} = sub { die; };

my $BINDIR = "$Bin/../build/bin";
my $SRC;
my $TRG;
my $OUTPUT = "lex";
my $THREADS = 8;
my $PARALLEL = 0;

GetOptions(
  "bindir=s" => \$BINDIR,
  "s|source=s" => \$SRC,
  "t|target=s" => \$TRG,
  "o|output=s" => \$OUTPUT,
  "threads=i" => \$THREADS,
  "parallel" => \$PARALLEL
);

die "--bindir arg is required" if not defined $BINDIR;
die "--source arg is required" if not defined $SRC;
die "--target arg is required" if not defined $TRG;
die "--output arg is required" if not defined $OUTPUT;

for my $app (qw(fast_align atools extract_lex)) {
    die "Could not find $app in $BINDIR" if not -e "$BINDIR/$app";
}

my $TEMPDIR = tempdir(CLEANUP => 1);

my (undef, $CORPUS) = tempfile( DIR => $TEMPDIR );
my (undef, $ALN_S2T) = tempfile( DIR => $TEMPDIR );
my (undef, $ALN_T2S) = tempfile( DIR => $TEMPDIR );
my (undef, $ALN_GDF) = tempfile( DIR => $TEMPDIR );

execute("paste $SRC $TRG | sed 's/\\t/ ||| /' > $CORPUS");

my @COMMANDS = (
    "OMP_NUM_THREADS=$THREADS $BINDIR/fast_align -vdo -i $CORPUS > $ALN_S2T",
    "OMP_NUM_THREADS=$THREADS $BINDIR/fast_align -vdor -i $CORPUS > $ALN_T2S"
);

my @PIDS;
for my $c (@COMMANDS) {
    if($PARALLEL) {
        my $pid = fork();
        if (!$pid) {
            execute($c);
            exit(0);
        }
        else {
            push(@PIDS, $pid);
            print "Forked process $pid\n";
        }
    }
    else {
        execute($c);
    }
}
if($PARALLEL) {
    waitpid($_, 0) foreach(@PIDS);
}

execute("$BINDIR/atools -c grow-diag-final -i $ALN_S2T -j $ALN_T2S > $ALN_GDF");
execute("$BINDIR/extract_lex $TRG $SRC $ALN_GDF $OUTPUT.s2t $OUTPUT.t2s");

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

sub logMessage {
    my $message = shift;
    my $time = POSIX::strftime("%m/%d/%Y %H:%M:%S", localtime());
    my $log_message = $time."\t$message\n";
    print STDERR $log_message;
}
