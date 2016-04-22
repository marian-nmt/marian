#!/usr/bin/perl

use strict;
use POSIX;

my $PID = $$;

my $MOSES = "/work/mosesdecoder/bin/moses";
my $RESCORER = "/work/amunn/build/bin/rescorer";
my $RESCORER_WRAPPER = "/work/amunn/scripts/rescore.pl";

my $NMT = "/work/wmt16/work/mjd.en-ru.penn/work.en-ru/nmt.ru-en";

my $MODELS = join(" ", map { "-m $NMT/$_" } qw(model.iter510000.npz model.iter540000.npz model.iter570000.npz));
my ($SVCB, $TVCB) = map { "$NMT/$_" } qw(vocab.ru vocab.en);
my $FEATURES = join(" ", map { "-f $_" } qw(N0 N1 N2));

for(my $i = 0; $i < @ARGV; $i++) {
  if($ARGV[$i] =~ /weight-overwrite/) {
    $ARGV[$i+1] = "'". $ARGV[$i+1] . "'";
  }
}

my $opts = join(" ", @ARGV);

my ($nbest) = $opts =~ /-n-best-list (run.*?.best100.out)/;

if($opts =~ /-show-weights/) {
    exec("$MOSES $opts");
}
else {
  $opts =~ /-input-file (\S+)/;
  my $input = $1;
  print STDERR "OPTS: $opts\n";
  execute("$MOSES $opts");
  execute("$RESCORER_WRAPPER -r $RESCORER $MODELS $FEATURES -s $SVCB -t $TVCB -n $nbest -i $input -w features.list > $nbest.temp");
  rename("$nbest.temp", $nbest);
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

sub logMessage {
    my $message = shift;
    my $time = POSIX::strftime("%m/%d/%Y %H:%M:%S", localtime());
    my $log_message = $time."\t$message\n"; 
    print STDERR $log_message;
}

 