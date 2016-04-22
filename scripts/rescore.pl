#!/usr/bin/env perl

use strict;
use Getopt::Long;
use File::Temp qw(tempfile);

my $RESCORER;
my $INPUT;
my $NBEST;
my $WEIGHTS;

my @MODELS;
my ($VSRC, $VTRG);
my @FEATURES;

GetOptions(
    "i|input=s" => \$INPUT,
    "n|n-best=s" => \$NBEST,
    "f|features=s" => \@FEATURES,
    "m|models=s" => \@MODELS,
    "s|source=s" => \$VSRC,
    "t|target=s" => \$VTRG,
    "r|rescorer=s" => \$RESCORER,
    "w|weights=s" => \$WEIGHTS
);

my $BEFORE = "LM1=";
open(W, "<", $WEIGHTS) or die "Could not open";
chomp(my $FIRST = <W>);
($BEFORE) = split(/\s/, $FIRST);
while (<W>) {
    my ($CURRENT) = split(/\s/, $_);
    print STDERR "$CURRENT\n";
    if ($CURRENT eq "$FEATURES[0]=") {
        print STDERR "Found $FEATURES[0] after $BEFORE\n";
        last;
    }
    $BEFORE = $CURRENT;
}
close(W);

my ($NBEST_TEMP_HANDLE, $NBEST_TEMP_FILE1) = tempfile();
my (undef, $NBEST_TEMP_FILE2) = tempfile();
open(NBEST_IN, "<", $NBEST) or die "Could not open";
while (<NBEST_IN>) {
    chomp;
    foreach my $name (@FEATURES) {
        s/$name= \S+ //g;
    }
    print $NBEST_TEMP_HANDLE $_, "\n";
}
close(NBEST_IN);
close($NBEST_TEMP_HANDLE);

foreach my $i (0 .. $#MODELS) {
    system("$RESCORER -i $INPUT -m $MODELS[$i] -s $VSRC -t $VTRG -f $FEATURES[$i] -n $NBEST_TEMP_FILE1 > $NBEST_TEMP_FILE2");
    rename($NBEST_TEMP_FILE2, $NBEST_TEMP_FILE1);
}

open($NBEST_TEMP_HANDLE, "<", $NBEST_TEMP_FILE1) or die "Could not open";

my $PATTERN1 = quotemeta(join(" ", map { "\\w$_= \\S+" } @FEATURES));
my $PATTERN2 = quotemeta("\\w$BEFORE \\S+");

while (<$NBEST_TEMP_HANDLE>) {
    chomp;
    if (/$PATTERN2/) {
        if(s/($PATTERN1)//) {
            my $FEAT = $1;
            s/($PATTERN2 )/$1$FEAT lala /;   
        }
    }
    print "$_\n";
}

