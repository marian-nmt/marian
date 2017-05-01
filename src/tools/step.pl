#!/usr/bin/perl

use strict;
use Data::Dumper;
use Algorithm::Diff::XS qw(traverse_balanced);
use Getopt::Long;

my $cdi;
my $pal;
GetOptions(
  "cdi"  => \$cdi,
  "pal"  => \$pal
) or die("Error in command line arguments\n");

$Data::Dumper::Indent = 0;

while(<STDIN>) {
  chomp;
  my ($src, $trg, $aln) = split(/\t/, $_);
  
  #print $src, "\n";
  #print $trg, "\n";
  
  my @src = split(/\s/, $src);
  my @trg = split(/\s/, $trg);
  
  #print scalar @src, " ", scalar @trg, "\n";

  push(@src, "</s>");
  push(@trg, "</s>");
  
  my @seq;
  
  if($cdi) {
    traverse_balanced(
        \@trg, \@src,
        {
            MATCH => sub { push(@seq, "<c>", $trg[$_[0]]); },
            DISCARD_A => sub { push(@seq, $trg[$_[0]]); },
            DISCARD_B => sub { push(@seq, "<d>"); },
            CHANGE    => sub { push(@seq, "<r>", $trg[$_[0]]); },
        }
    );      
  }
  elsif($pal) {
    traverse_balanced(
        \@trg, \@src,
        {
            MATCH => sub { push(@seq, "$_[1]-$_[0]"); },
            DISCARD_A => sub { push(@seq, "$_[1]-$_[0]"); },
            DISCARD_B => sub { },
            CHANGE    => sub { push(@seq, "$_[1]-$_[0]"); },
        }
    );
    @seq = sort { $a <=> $b } @seq;
  }
  else {
    traverse_balanced(
        \@trg, \@src,
        {
            MATCH => sub { push(@seq, "<step>", $trg[$_[0]]); },
            DISCARD_A => sub { push(@seq, $trg[$_[0]]); },
            DISCARD_B => sub { push(@seq, "<step>"); },
            CHANGE    => sub { push(@seq, "<step>", $trg[$_[0]]); },
        }
    ); 
  }
  
  if(not $pal) {
    shift(@seq);
  }
  pop(@seq);
  print join(" ", @seq), "\n";
}